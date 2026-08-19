"""教学版 Triton FlashAttention v1 前向算子。

这个文件刻意只实现 baseline 需要的 forward：FP16/BF16、causal/non-causal、
head_dim <= 128。每个 Triton program 负责一个 ``(query block, head)``，沿 K/V
方向分块扫描，并用 online softmax 保存行最大值 ``m_i`` 和归一化因子 ``l_i``，
因此不会物化完整的 ``Q @ K.T`` 注意力矩阵。

连续 KV 接口的输入布局为 ``[sequence, num_heads, head_dim]``。此外，本文件
还提供直接消费扁平 Q、分页 KV cache 和变长 batch metadata 的接口；分页路径
不会在 kernel 外 gather 历史 K/V。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:  # CPU 开发机可以导入项目；真正 launch 时才要求安装 Triton。
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 由 CPU fallback 覆盖
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _flash_attention_v1_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        q_len,
        kv_len,
        softmax_scale_log2,
        causal_offset,
        stride_qm: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """FlashAttention v1 的 online-softmax 主循环。

        q/k/v 不必为方阵：decode 时 q_len=1、kv_len=context+1，
        ``causal_offset`` 表示第 0 个 query 在完整序列中的绝对位置。
        """

        query_block = tl.program_id(0)
        head = tl.program_id(1)

        offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM)
        q = tl.load(
            q_ptr
            + offs_m[:, None] * stride_qm
            + head * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        # V1 的核心状态：每一行只保留 running max、running sum 和输出累加器。
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, kv_len, BLOCK_N):
            key_pos = start_n + offs_n
            k_mask = (offs_d[:, None] < HEAD_DIM) & (key_pos[None, :] < kv_len)
            k = tl.load(
                k_ptr
                + key_pos[None, :] * stride_kn
                + head * stride_kh
                + offs_d[:, None] * stride_kd,
                mask=k_mask,
                other=0.0,
            )
            scores = tl.dot(q, k) * softmax_scale_log2
            valid = (offs_m[:, None] < q_len) & (key_pos[None, :] < kv_len)
            if IS_CAUSAL:
                valid &= key_pos[None, :] <= (
                    causal_offset + offs_m[:, None]
                )
            scores = tl.where(valid, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp2(m_i - m_new)
            probabilities = tl.exp2(scores - m_new[:, None])
            l_new = l_i * alpha + tl.sum(probabilities, axis=1)

            v_mask = (key_pos[:, None] < kv_len) & (
                offs_d[None, :] < HEAD_DIM
            )
            v = tl.load(
                v_ptr
                + key_pos[:, None] * stride_vn
                + head * stride_vh
                + offs_d[None, :] * stride_vd,
                mask=v_mask,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        output = acc / l_i[:, None]
        out_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM)
        tl.store(
            out_ptr
            + offs_m[:, None] * stride_om
            + head * stride_oh
            + offs_d[None, :] * stride_od,
            output,
            mask=out_mask,
        )

    @triton.jit
    def _paged_varlen_flash_attention_v1_fwd_kernel(
        q_ptr,
        kv_cache_ptr,
        block_table_ptr,
        query_start_loc_ptr,
        seq_lens_ptr,
        out_ptr,
        max_seq_len,
        softmax_scale_log2,
        stride_qm: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_cache_block: tl.constexpr,
        stride_cache_kv: tl.constexpr,
        stride_cache_token: tl.constexpr,
        stride_cache_head: tl.constexpr,
        stride_cache_d: tl.constexpr,
        stride_bt_req: tl.constexpr,
        stride_bt_block: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        Q_HEADS_PER_KV_HEAD: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """分页 KV + 变长 batch 的 online-softmax 主循环。

        grid 三个维度分别表示 ``Q tile / Q head / request``。一个 program 只
        负责一个 request 的一个 Q head，因而不同请求之间天然隔离。
        """

        query_block = tl.program_id(0)
        query_head = tl.program_id(1)
        request_id = tl.program_id(2)

        # 【变长Batch优化】每个 program 通过 query_start_loc 找到自己请求在
        # 扁平 Q 中的边界。矩形 grid 中超过真实 query_len 的 tile 全程 mask。
        request_q_start = tl.load(query_start_loc_ptr + request_id)
        request_q_end = tl.load(query_start_loc_ptr + request_id + 1)
        query_len = request_q_end - request_q_start
        tile_q_start = query_block * BLOCK_M
        seq_len = tl.load(seq_lens_ptr + request_id)
        context_len = seq_len - query_len
        local_q_pos = tile_q_start + tl.arange(0, BLOCK_M)
        flat_q_pos = request_q_start + local_q_pos
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = (local_q_pos[:, None] < query_len) & (
            offs_d[None, :] < HEAD_DIM
        )
        q = tl.load(
            q_ptr
            + flat_q_pos[:, None] * stride_qm
            + query_head * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        # 【GQA优化】Qwen2.5 的 cache 只保存真实 KV heads。多个 Q heads 通过
        # 整数映射直接共享一个 KV head，不再 repeat_interleave 复制 KV 张量。
        kv_head = query_head // Q_HEADS_PER_KV_HEAD
        offs_n = tl.arange(0, BLOCK_N)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, max_seq_len, BLOCK_N):
            key_pos = start_n + offs_n
            key_valid = key_pos < seq_len
            logical_block = key_pos // BLOCK_SIZE
            block_offset = key_pos % BLOCK_SIZE

            # 【PagedAttention优化】页表翻译被合并进 attention kernel：不再
            # gather 连续历史 KV。每个 key token 在使用时才解析物理 block。
            physical_block = tl.load(
                block_table_ptr
                + request_id * stride_bt_req
                + logical_block * stride_bt_block,
                mask=key_valid,
                other=0,
            )

            # 【Decode 2D向量化加载优化】一次构造 BLOCK_N x BLOCK_D 地址矩阵，
            # 沿 token/head_dim 两维加载一个分页 K/V tile。K 用 [D,N] 供 q@k，
            # V 用 [N,D] 供 p@v；数据直接进入本 program 的片上工作集。
            k_mask = (offs_d[:, None] < HEAD_DIM) & key_valid[None, :]
            k = tl.load(
                kv_cache_ptr
                + physical_block[None, :] * stride_cache_block
                + block_offset[None, :] * stride_cache_token
                + kv_head * stride_cache_head
                + offs_d[:, None] * stride_cache_d,
                mask=k_mask,
                other=0.0,
            )
            scores = tl.dot(q, k) * softmax_scale_log2
            valid = (local_q_pos[:, None] < query_len) & key_valid[None, :]
            if IS_CAUSAL:
                # 【Prefill因果掩码】chunked prefill 的第一个 Q 对应 context_len；
                # decode 时 query_len=1，它自然可以看到 seq_len 内全部 KV。
                valid &= key_pos[None, :] <= (
                    context_len + local_q_pos[:, None]
                )
            scores = tl.where(valid, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp2(m_i - m_new)
            probabilities = tl.exp2(scores - m_new[:, None])
            l_new = l_i * alpha + tl.sum(probabilities, axis=1)

            v_mask = key_valid[:, None] & (offs_d[None, :] < HEAD_DIM)
            v = tl.load(
                kv_cache_ptr
                + physical_block[:, None] * stride_cache_block
                + stride_cache_kv
                + block_offset[:, None] * stride_cache_token
                + kv_head * stride_cache_head
                + offs_d[None, :] * stride_cache_d,
                mask=v_mask,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        output = acc / l_i[:, None]
        out_mask = (local_q_pos[:, None] < query_len) & (
            offs_d[None, :] < HEAD_DIM
        )
        tl.store(
            out_ptr
            + flat_q_pos[:, None] * stride_om
            + query_head * stride_oh
            + offs_d[None, :] * stride_od,
            output,
            mask=out_mask,
        )


def _torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    causal_offset: int,
) -> torch.Tensor:
    """CPU/float32 fallback，也作为 Triton kernel 的正确性参考。"""

    q = query.transpose(0, 1)
    k = key.transpose(0, 1)
    v = value.transpose(0, 1)
    if causal:
        q_pos = causal_offset + torch.arange(query.shape[0], device=query.device)
        kv_pos = torch.arange(key.shape[0], device=query.device)
        mask = kv_pos[None, :] <= q_pos[:, None]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        output = F.scaled_dot_product_attention(q, k, v)
    return output.transpose(0, 1).contiguous()


def flash_attention_v1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    causal_offset: int = 0,
) -> torch.Tensor:
    """执行 FlashAttention v1 forward。

    Args:
        query: ``[query_len, num_heads, head_dim]``。
        key/value: ``[kv_len, num_heads, head_dim]``。
        causal_offset: query[0] 在完整 K/V 序列中的位置。prefill 为历史长度，
            decode 通常等于 ``kv_len - 1``。
    """

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value 必须是 [seq, num_heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key 和 value 的形状必须相同")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("query/key/value 的 head 数和 head_dim 必须相同")
    if query.shape[0] == 0 or key.shape[0] == 0:
        raise ValueError("FlashAttention 不接受空序列")
    head_dim = query.shape[-1]
    if head_dim > 128:
        raise NotImplementedError("教学版 Triton FlashAttention 仅支持 head_dim <= 128")

    use_triton = (
        triton is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
    )
    if not use_triton:
        return _torch_attention(
            query, key, value, causal=causal, causal_offset=causal_offset
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.empty_like(query)
    block_d = triton.next_power_of_2(head_dim)
    block_m = 32
    block_n = 32
    grid = (triton.cdiv(query.shape[0], block_m), query.shape[1])
    _flash_attention_v1_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        query.shape[0],
        key.shape[0],
        1.0 / math.sqrt(head_dim) * math.log2(math.e),
        causal_offset,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return output


def _torch_paged_varlen_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    """分页变长接口的 CPU/非 Triton 正确性参考。"""

    output = torch.empty_like(query)
    num_q_heads = query.shape[1]
    num_kv_heads = kv_cache.shape[3]
    repeats = num_q_heads // num_kv_heads
    for request_id in range(seq_lens.numel()):
        q_start = int(query_start_loc[request_id].item())
        q_end = int(query_start_loc[request_id + 1].item())
        seq_len = int(seq_lens[request_id].item())
        query_len = q_end - q_start
        positions = torch.arange(seq_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        block_offsets = positions % block_size
        physical_blocks = block_table[request_id, logical_blocks].long()
        key = kv_cache[physical_blocks, 0, block_offsets]
        value = kv_cache[physical_blocks, 1, block_offsets]
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        output[q_start:q_end] = _torch_attention(
            query[q_start:q_end],
            key,
            value,
            causal=causal,
            causal_offset=seq_len - query_len,
        )
    return output


def paged_varlen_flash_attention_v1(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int,
    max_query_len: int,
    max_seq_len: int,
    causal: bool = True,
) -> torch.Tensor:
    """直接从分页 KV cache 计算扁平变长 batch 的 FlashAttention。

    Args:
        query: 所有请求拼接后的 ``[total_query_tokens, q_heads, head_dim]``。
        kv_cache: ``[num_blocks, 2, block_size, kv_heads, head_dim]``。
        block_table: ``[num_reqs, max_blocks_per_req]`` 物理页表。
        query_start_loc: 请求在扁平 query 中的前缀和边界。
        seq_lens: 写入本轮 K/V 后各请求的完整序列长度。

    该接口一次 kernel launch 覆盖整个变长 batch。它没有外部 KV gather；
    GQA 通过 head 索引映射共享真实 KV heads，不会把 4 个 KV heads 复制成
    28 份。不同 Q heads 仍由独立 program 计算。
    """

    if query.ndim != 3:
        raise ValueError("query 必须是 [total_tokens, q_heads, head_dim]")
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise ValueError(
            "kv_cache 必须是 [num_blocks, 2, block_size, kv_heads, head_dim]"
        )
    if kv_cache.shape[2] != block_size:
        raise ValueError("kv_cache 的 block_size 与 metadata 不一致")
    if query.shape[-1] != kv_cache.shape[-1]:
        raise ValueError("Q 和 KV cache 的 head_dim 必须相同")
    if query.shape[1] % kv_cache.shape[3]:
        raise ValueError("q_heads 必须能被 kv_heads 整除")
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("query_start_loc 和 seq_lens 必须是一维张量")
    if query_start_loc.numel() != seq_lens.numel() + 1:
        raise ValueError("query_start_loc 长度必须等于 num_reqs + 1")
    if block_table.ndim != 2 or block_table.shape[0] != seq_lens.numel():
        raise ValueError("block_table 必须是 [num_reqs, max_blocks_per_req]")
    if query.shape[0] == 0 or seq_lens.numel() == 0:
        raise ValueError("PagedAttention 不接受空 batch")
    if max_query_len <= 0 or max_seq_len <= 0:
        raise ValueError("max_query_len/max_seq_len 必须为正数")
    head_dim = query.shape[-1]
    if head_dim > 128:
        raise NotImplementedError("教学版 Triton FlashAttention 仅支持 head_dim <= 128")

    use_triton = (
        triton is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and kv_cache.dtype == query.dtype
        and block_table.is_cuda
        and query_start_loc.is_cuda
        and seq_lens.is_cuda
    )
    if not use_triton:
        return _torch_paged_varlen_attention(
            query,
            kv_cache,
            block_table,
            query_start_loc,
            seq_lens,
            block_size=block_size,
            causal=causal,
        )

    query = query.contiguous()
    output = torch.empty_like(query)
    block_d = triton.next_power_of_2(head_dim)
    block_m = 32
    block_n = 32
    grid = (
        triton.cdiv(max_query_len, block_m),
        query.shape[1],
        seq_lens.numel(),
    )
    _paged_varlen_flash_attention_v1_fwd_kernel[grid](
        query,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        output,
        max_seq_len,
        1.0 / math.sqrt(head_dim) * math.log2(math.e),
        *query.stride(),
        *kv_cache.stride(),
        *block_table.stride(),
        *output.stride(),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        Q_HEADS_PER_KV_HEAD=query.shape[1] // kv_cache.shape[3],
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return output
