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
from typing import Optional, Tuple
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


