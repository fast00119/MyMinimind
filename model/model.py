from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers.activations import ACT2FN

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nn.parameter(torch.ones(dim))

    def _norm(self, x: torch.tensor):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps) * x

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_angles(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0/(rope_base ** (torch.arange(0,dim,2)[:dim//2].float()/dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        # 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast : 高频边界，波长比例大于此值的维度不缩放
        # beta_slow : 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",16),
            rope_scaling.get("beta_fast",32),
            rope_scaling.get("beta_slow",1),
            rope_scaling.get("attention_factor",1.0)
        )

        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max/(2*b*math.pi))) / (2*math.log(rope_base))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim//2-1)

            ramp = torch.clamp((torch.arange(dim//2, device=freqs.device).float()-low) 
                    / max((high-low), 0.001), 0, 1)

            freqs = freqs * (1 - ramp + ramp/factor)

    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()

    angles_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
    angles_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor

    return angles_cos, angles_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotary_half(x):
        return torch.cat([
            -x[..., x.shape[-1]:],
            x[..., :x.shape[-1]]
        ], dim=-1)

    q_embed = q * cos.unsqueeze(unsqueeze_dim) + rotary_half(q) * sin.unsqueeze(unsqueeze_dim)
    k_embed = k * cos.unsqueeze(unsqueeze_dim) + rotary_half(k) * sin.unsqueeze(unsqueeze_dim)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复key-value张量以匹配query头数 (用于分组查询注意力GQA)
    在GQA中，key和value的头数少于query，需要重复来匹配
    例如：8个query头，2个kv头，则需要每个kv头重复4次
    
    Args:
        x: kv张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的张量 [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x

    bs, seq_len, num_kv_heads, head_dim = x.shape
    return x.unsqueeze(3).expand(bs, seq_len, num_kv_heads, n_rep, head_dim)\
        .reshape(bs, seq_len, num_kv_heads*n_rep, head_dim)

class Attention(nn.Module):
    """
    多头自注意力机制，支持分组查询注意力(GQA)和Flash Attention优化
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()

        assert args.num_attention_heads % args.num_key_value_heads == 0

        # 注意力头配置
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        # linear projection
        self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads*self.head_dim, args.hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检查是否支持 Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) 其中cos 的shape为 [end, head_dim]
            past_key_value: (past_k, past_v) | None   
            其中past_k 的shape为 [batch_size, past_seq_len, n_local_kv_heads, head_dim]
            use_cache: 是否使用kv cache
            attention mask: 注意力掩码，掩盖pad位置 [batch_size, seq_len]
        
        Returns:
            output: 输出张量 [batch_size, seq_len, hidden_size]
            past_kv: 更新后的kv cache (past_k, past_v)
        """
        bs, seq_len, _ = x.shape

        # linear projection
        xq = self.q_proj(x).view(bs, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bs, seq_len, self.n_local_kv_heads, self.head_dim)

        # RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # apply kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1,2)
        xk = repeat_kv(xk, self.n_rep).transpose(1,2)
        xv = repeat_kv(xv, self.n_rep).transpose(1,2)

        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = None if attention_mask is None else attention_mask.view(bs, 1, 1, -1).expand(bs, self.n_local_heads, seq_len, -1).bool()
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(self.head_dim)

            causal_mask = torch.triu(torch.full((seq_len,seq_len), float("-inf")), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores += extended_attention_mask

        weight = torch.softmax(scores.float(), dim=-1).type_as(xq)
        weight = self.attn_dropout(weight)

        output = self.o_proj((weight @ xv).transpose(1,2).reshape(bs, seq_len, -1))
        output = self.resid_dropout(output)

        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
                intermediate_size = int(args.hidden_size * 8 / 3)
                args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))

class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.router = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.router.weight, a = math.sqrt(5))

    def forward(self, hidden_states):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            topk_idx: [batch_size*seq_len, topk]
            topk_weight: [batch_size*seq_len, topk]
            aux_loss: scalar
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = self.router(hidden_states)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=1, sorted=False)

        if self.top_k>1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha>0.0:
            scores_for_aux = scores
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 每个sequence单独计算专家负载，再在batch上取平均
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones(bsz, seq_len*self.top_k, device=hidden_states.device)
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 把整个batch所有token放在一起统计专家负载
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    
class MoEFeedFroward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # 专家层
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )
        # 门控层
        self.gate = MoEGate(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        topk_idx, topk_weight, aux_loss = self.gate(x)
        self.aux_loss = aux_loss
        x = x.view(-1, h)
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            for i, expert in enumerate(self.experts):
                expert_output = expert(x[flat_topk_idx == i])
                if expert_output.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_output.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_output.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和，y表示每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1)).view(
                *orig_shape
            )
        # 计算共享专家输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        Args:
            x: [batch_size * seq_len, hidden_size]
            flat_expert_indices: batch_size * seq_len * topk
            flat_expert_weights: batch_size * seq_len * topk
        Returns:
            expert_cache: [batch_size * seq_len, hidden_size]
        """
        flat_expert_weights = flat_expert_weights.unsqueeze(-1)
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )
        
        return expert_cache

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id

        self.attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedFroward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # GQA       
        residual = hidden_states
        hidden_states, present_key_value = self.attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = hidden_states + residual

        # FFN
        hidden_states = hidden_states + self.mlp(self.post_attn_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(idx, config) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        angles_cos, angles_sin = precompute_angles(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling
        )
        self.register_buffer("angles_cos", angles_cos, persistent=True)
        self.register_buffer("angles_sin", angles_sin, persistent=True)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: 每层的kv cache列表 past_key_values[0] 形如(past_k, past_v)\
            其中 past_k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
            use_cache: 是否使用 kv cache
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
            presents: 更新后每层的kv cache
            aux_loss: 所有层MoEFeedForward的aux_loss之和
        """
        bsz, seq_len = input_ids.shape

        # 兼容性检查
        if hasattr(past_key_values, 'layers'):
                    past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算start_pos: 已有past序列的长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeding = (
            self.angles_cos[start_pos: start_pos+seq_len],
            self.angles_sin[start_pos: start_pos+seq_len]
        )

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeding,
                past_key_value,
                use_cache,
                attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp,MoEFeedFroward)
        )
        return hidden_states, presents, aux_loss
    
