import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# 打印日志
def Logger(content):
    if is_main_process():
        print(content)

# 余弦退火学习率
def get_lr(current_step, total_steps, lr):
    return (lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))))

# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 保存或读取检查点，用于断点续训
def lm_checkpoint(
    lm_config,
    weight="pretrain",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="../checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        # 保存模式，模型权重保存到ckp_path，完整训练状态保存到resume_path
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model":state_dict,
            "optimizer":optimizer.state_dict(),
            "epoch":epoch,
            "step":step,
            "world_size":dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id":wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:
        # 加载模式，从resume_path加载训练状态
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")

            return ckp_data
        return None

# 初始化模型和分词器
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindForCausalLM

    # 如果没有指定tokenizer_path，使用项目根目录下的model文件夹
    if tokenizer_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    model = MiniMindForCausalLM(lm_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if from_weight != "none":
        moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_path}.pth"

        weight = torch.load(weight_path, map_location=device)
        model.load_state_dict(weight, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.6f} m")

    return model.to(device), tokenizer

# 初始化分布式
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0 # 非DDP模式

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(f"cuda:{local_rank}")
    return local_rank

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []

        if len(batch)>0 and skipped>=self.skip_batches:
            yield batch

    def __len__(self):
        total_batches =  math.ceil(len(self.sampler)/self.batch_size)
        return max(0, total_batches-self.skip_batches)