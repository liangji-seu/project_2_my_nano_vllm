"""Hugging Face 权重兼容、支持 TP/PP 的 Qwen2 推理模型。

实现刻意保留 Hugging Face 参数名，便于把 checkpoint 参数直接映射到局部
shard；执行结构则遵循 vLLM 的模型并行语义：QKV/gate/up 按输出维切分，
o/down 按输入维切分并 all-reduce，embedding/lm_head 按 vocabulary 切分，
decoder layers 按 PP stage 连续分段。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_vllm.attention.triton_flash_attention import (
    flash_attention_v1,
    paged_varlen_flash_attention_v1,
)
from my_vllm.attention.metadata import (
    FullAttentionMetadata,
    FullAttentionMetadataCollection,
)
from my_vllm.distributed.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
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


def _divide(value: int, divisor: int, name: str) -> int:
    if value % divisor:
        raise ValueError(f"{name}={value} 必须能被 TP={divisor} 整除")
    return value // divisor


class ColumnParallelLinear(nn.Module):
    """【TP·列并行】保存输出维 shard，forward 不做集合通信。"""

    def __init__(self, input_size: int, output_size: int, *, bias: bool):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = _divide(
            output_size, self.tp_size, "linear output_size"
        )
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.output_size_per_partition))
            if bias
            else None
        )
        self.reset_parameters(input_size)

    def reset_parameters(self, input_size: int) -> None:
        # 与 torch.nn.Linear 的默认初始化一致；meta 构造时该调用不分配内存，
        # checkpoint loader 随后仍会替换成真实 shard。
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(ColumnParallelLinear):
    """【TP·QKV packed】把 HF 的 Q/K/V shard 合并为一次局部 GEMM。"""

    def __init__(
        self,
        hidden_size: int,
        total_q_size: int,
        total_kv_size: int,
        *,
        bias: bool,
    ):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_sizes_per_partition = (
            _divide(total_q_size, self.tp_size, "Q projection size"),
            _divide(total_kv_size, self.tp_size, "K projection size"),
            _divide(total_kv_size, self.tp_size, "V projection size"),
        )
        super().__init__(
            hidden_size,
            total_q_size + 2 * total_kv_size,
            bias=bias,
        )


class MergedColumnParallelLinear(ColumnParallelLinear):
    """【TP·Merged Column】把 gate/up shard 合并为一次局部 GEMM。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: tuple[int, ...],
        *,
        bias: bool,
    ):
        tp_size = get_tensor_model_parallel_world_size()
        self.output_sizes_per_partition = tuple(
            _divide(size, tp_size, "merged projection size")
            for size in output_sizes
        )
        super().__init__(input_size, sum(output_sizes), bias=bias)


class RowParallelLinear(nn.Module):
    """【TP·行并行】消费输入维 shard，局部 GEMM 后执行 SUM all-reduce。"""

    def __init__(self, input_size: int, output_size: int, *, bias: bool = False):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = _divide(
            input_size, self.tp_size, "linear input_size"
        )
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_size_per_partition)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, None)
        tensor_model_parallel_all_reduce(output)
        # bias 只能在 reduce 后加一次，否则会被 TP SUM 放大。
        if self.bias is not None:
            output = output + self.bias
        return output


class VocabParallelEmbedding(nn.Module):
    """【TP·词表并行】每个 rank 只保存连续 vocabulary shard。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_embeddings_per_partition = _divide(
            vocab_size, self.tp_size, "vocab_size"
        )
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, hidden_size)
        )
        nn.init.normal_(self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outside = (input_ids < self.vocab_start_index) | (
            input_ids >= self.vocab_end_index
        )
        local_ids = (input_ids - self.vocab_start_index).masked_fill(outside, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(outside.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(output)


class ParallelLMHead(ColumnParallelLinear):
    """【TP·并行 LM Head】局部 vocabulary projection 后收集完整 logits。"""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = super().forward(hidden_states)
        return tensor_model_parallel_all_gather(local_logits, dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = _divide(
            config.num_attention_heads, self.tp_size, "num_attention_heads"
        )
        self.num_kv_heads = _divide(
            config.num_key_value_heads, self.tp_size, "num_key_value_heads"
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 明确区分两个空间：hidden_size 是模型主干中每个 token 的维度；
        # q_size 是全部 Q heads（以及全部 attention 输出 heads）拼接后的维度。
        # Qwen2.5 中二者数值恰好相同，但语义并不相同。
        self.total_q_size = config.num_attention_heads * self.head_dim
        self.total_kv_size = config.num_key_value_heads * self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        if self.total_q_size != config.hidden_size:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        self.rope_theta = config.rope_theta
        bias = config.attention_bias
        self.qkv_proj = QKVParallelLinear(
            # hidden_size 是 token 语义空间；Q/K/V 输出在 TP rank 内按 head 分片。
            config.hidden_size,
            self.total_q_size,
            self.total_kv_size,
            bias=bias,
        )
        self.o_proj = RowParallelLinear(
            # 输入是所有 attention heads 的输出拼接空间，输出才回到
            # token 的 hidden/residual 语义空间。
            self.total_q_size,
            config.hidden_size,
            bias=False,
        )
        # initialize_kv_cache 后由 runner 绑定，布局为
        # [num_blocks, 2(K/V), block_size, num_kv_heads, head_dim]。
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
        """扁平 Qwen2 attention：写分页 KV，再执行变长 FlashAttention。

        ``slot_mapping`` 决定当前 K/V 写到哪个物理 slot；``block_table`` 把
        每个请求的逻辑 block 翻译成物理 block。PagedAttention 路径直接在
        Triton kernel 内完成页表翻译，不再逐请求 gather 连续历史 K/V。
        """

        if hidden_states.ndim != 2:
            raise ValueError("Qwen2Attention 只接受 [total_tokens, hidden_size]")
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )
        q = q.view(
            num_tokens, self.num_heads, self.head_dim
        )
        k = k.view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v = v.view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        # 给batch的输入token向量对应的 q向量 + k向量，增加位置编码
        q, k = self._apply_rope(q, k, positions)


        if attention_metadata is None:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            output = flash_attention_v1(q, k, v, causal=True)
        elif self.kv_cache is None:
            # 独立模型单测/显存 profiling 没有初始化 KV cache。仍使用扁平
            # block-diagonal attention，确保不同 request 绝不互相看到 token。
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
                q.transpose(0, 1),
                k.repeat_interleave(
                    self.num_heads // self.num_kv_heads, dim=1
                ).transpose(0, 1),
                v.repeat_interleave(
                    self.num_heads // self.num_kv_heads, dim=1
                ).transpose(0, 1),
                attn_mask=attention_mask,
                is_causal=False,
            ).transpose(0, 1)
        else:
            if attention_metadata.num_actual_tokens != num_tokens:
                raise ValueError("metadata token 数与 hidden_states 不一致")


            # 把我们输入token向量的k,v向量，全部写入我们的kvcache tensor
            self._write_to_kv_cache(k, v, attention_metadata.slot_mapping)

            '''
            所以，一次flashattention的算子调用，其实是一口气批量处理整个batch的q向量
            这个算子，其实就是内核的wrapper

            所以，层 包含如何调用算子， 算子 = 核函数的wrapper
            '''
            output = paged_varlen_flash_attention_v1(
                q, # 一个batch的所有token的q向量， shape = (num_tokens, self.num_heads, self.head_dim)
                self.kv_cache, # 这个layer的kvcache显存张量，也就是我们的物理显存

                # 注意力元数据，包含block_table, 位置索引
                attention_metadata.block_table, # 块 页表
                attention_metadata.query_start_loc,
                attention_metadata.seq_lens,
                block_size=attention_metadata.block_size,
                max_query_len=attention_metadata.max_query_len,
                max_seq_len=attention_metadata.max_seq_len,
                causal=attention_metadata.causal,
            )
        output = output.contiguous().view(num_tokens, -1)
        return self.o_proj(output)

    def _write_to_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """按照线性 slot 地址把当前 token 的 K/V scatter 到物理页。"""

        assert self.kv_cache is not None
        block_size = self.kv_cache.shape[2]
        physical_blocks = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_offsets = slot_mapping % block_size
        self.kv_cache[physical_blocks, 0, block_offsets] = key.to(
            self.kv_cache.dtype
        )
        self.kv_cache[physical_blocks, 1, block_offsets] = value.to(
            self.kv_cache.dtype
        )

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        if config.hidden_act not in ("silu", "swish"):
            raise ValueError(f"暂不支持激活函数 {config.hidden_act}")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


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
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()
        self.start_layer = config.num_hidden_layers * self.pp_rank // self.pp_size
        self.end_layer = (
            config.num_hidden_layers * (self.pp_rank + 1) // self.pp_size
        )
        # ModuleDict 的数字 key 会保留 HF 的 ``model.layers.<global_idx>`` 名字，
        # 同时不为不属于本 stage 的 layer 建立任何参数。
        self.layers = nn.ModuleDict(
            {
                str(index): Qwen2DecoderLayer(config)
                for index in range(self.start_layer, self.end_layer)
            }
        )
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if is_pipeline_first_stage()
            else None
        )
        self.norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if is_pipeline_last_stage()
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        attention_metadata: FullAttentionMetadataCollection | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_pipeline_first_stage():
            if input_ids is None or input_ids.ndim != 1:
                raise ValueError("第一个 PP stage 需要扁平 input_ids")
            assert self.embed_tokens is not None
            hidden_states = self.embed_tokens(input_ids)
        elif hidden_states is None or hidden_states.ndim != 2:
            raise ValueError("非首 PP stage 需要 [total_tokens, hidden_size] activation")
        assert hidden_states is not None
        if positions is None:
            positions = torch.arange(
                hidden_states.shape[0], device=hidden_states.device, dtype=torch.int64
            )
        if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[0]:
            raise ValueError("positions 必须与扁平 token/activation 数量一致")
        for global_index, layer in self.layers.items():
            layer_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.for_layer(
                    f"model.layers.{global_index}.self_attn"
                )
            )
            hidden_states = layer(hidden_states, positions, layer_metadata)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        if config.tie_word_embeddings and get_pipeline_model_parallel_world_size() > 1:
            raise NotImplementedError("PP>1 暂不支持跨 stage tied embeddings")
        self.lm_head = (
            ParallelLMHead(config.hidden_size, config.vocab_size)
            if is_pipeline_last_stage()
            else None
        )
        if config.tie_word_embeddings:
            assert self.lm_head is not None and self.model.embed_tokens is not None
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        attention_metadata: FullAttentionMetadataCollection | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 与 vLLM 模型接口一致：forward 返回 hidden states，logits 单独计算，
        # 避免为 batch 内每个 token 都物化 vocab-size logits。
        return self.model(
            input_ids,
            positions,
            attention_metadata,
            hidden_states=hidden_states,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("只有最后一个 PP stage 能计算 logits")
        return self.lm_head(hidden_states)

    def attention_layers(self):
        for global_index, layer in self.model.layers.items():
            yield f"model.layers.{global_index}.self_attn", layer.self_attn
