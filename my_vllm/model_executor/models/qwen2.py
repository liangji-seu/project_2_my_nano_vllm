"""可读性优先的 Hugging Face 兼容 Qwen2 模型。

这里暂不做 vLLM 的 QKV/MLP packed 参数和 TP 切分，参数名保持 Hugging Face
原样，目的是把「config 构造参数树 -> checkpoint 名字递归匹配」先做成闭环。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.attention.metadata import (
    FullAttentionMetadata,
    FullAttentionMetadataCollection,
)


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen2Config":
        return cls(
            **{
                name: data[name]
                for name in cls.__dataclass_fields__
                if name in data
            }
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        self.rope_theta = config.rope_theta
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # initialize_kv_cache 后由 runner 绑定；forward/InputBatch 暂不消费它。
        self.kv_cache: torch.Tensor | None = None

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对扁平 token 使用外部绝对 positions 计算 RoPE。"""

        if positions.ndim != 1 or positions.shape[0] != q.shape[0]:
            raise ValueError("positions 必须是与扁平 token 一一对应的一维张量")
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, device=q.device, dtype=torch.float32
                )
                / self.head_dim
            )
        )
        freqs = torch.outer(positions.to(torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[:, None, :]
        cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: FullAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """基础 PyTorch FullAttention，只处理完整的当前 prefill。

        输入始终是 ``[total_num_tokens, hidden_size]``。当 metadata 包含多个
        请求时，用 block-diagonal causal mask 隔离各请求。历史 KV 的分页读写
        属于下一阶段；因此 context_lens 非零时显式拒绝执行。
        """

        if hidden_states.ndim != 2:
            raise ValueError("Qwen2Attention 只接受 [total_tokens, hidden_size]")
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(
            num_tokens, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        q, k = self._apply_rope(q, k, positions)
        repeats = self.num_heads // self.num_kv_heads
        if repeats != 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        q = q.transpose(0, 1)  # [num_heads, total_tokens, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if attention_metadata is None:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if attention_metadata.num_actual_tokens != num_tokens:
                raise ValueError("metadata token 数与 hidden_states 不一致")
            if bool((attention_metadata.context_lens != 0).any().item()):
                raise NotImplementedError(
                    "基础 FullAttention 尚未读取历史 KV；"
                    "num_computed_tokens > 0 需要下一阶段的 KV Cache Attention"
                )
            req_indices = torch.repeat_interleave(
                torch.arange(
                    attention_metadata.num_reqs,
                    device=hidden_states.device,
                    dtype=torch.int64,
                ),
                attention_metadata.query_lens.to(torch.int64),
            )
            if req_indices.numel() != num_tokens:
                raise ValueError("query_lens 之和与扁平 token 数不一致")
            same_request = req_indices[:, None] == req_indices[None, :]
            causal = positions[:, None] >= positions[None, :]
            attention_mask = same_request & causal
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=False,
            )
        output = output.transpose(0, 1).contiguous().view(num_tokens, -1)
        return self.o_proj(output)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError(f"暂不支持激活函数 {config.hidden_act}")
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: FullAttentionMetadata | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), positions, attention_metadata
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: FullAttentionMetadataCollection | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("Qwen2Model 只接受扁平 input_ids: [total_num_tokens]")
        if positions is None:
            positions = torch.arange(
                input_ids.shape[0], device=input_ids.device, dtype=torch.int64
            )
        if positions.shape != input_ids.shape:
            raise ValueError("positions 必须与 input_ids 形状一致")
        hidden_states = self.embed_tokens(input_ids)
        for index, layer in enumerate(self.layers):
            layer_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.for_layer(
                    f"model.layers.{index}.self_attn"
                )
            )
            hidden_states = layer(hidden_states, positions, layer_metadata)
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: FullAttentionMetadataCollection | None = None,
    ) -> torch.Tensor:
        # 与 vLLM 模型接口一致：forward 返回 hidden states，logits 单独计算，
        # 避免为 batch 内每个 token 都物化 vocab-size logits。
        return self.model(input_ids, positions, attention_metadata)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def attention_layers(self):
        for index, layer in enumerate(self.model.layers):
            yield f"model.layers.{index}.self_attn", layer.self_attn
