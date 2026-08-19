"""教学版 Triton FlashAttention v1 前向算子。

这个文件刻意只实现 baseline 需要的 forward：FP16/BF16、causal/non-causal、
head_dim <= 128。每个 Triton program 负责一个 ``(query block, head)``，沿 K/V
方向分块扫描，并用 online softmax 保存行最大值 ``m_i`` 和归一化因子 ``l_i``，
因此不会物化完整的 ``Q @ K.T`` 注意力矩阵。

输入布局统一为 ``[sequence, num_heads, head_dim]``。变长 batch 在更上层按照
request 切片；这让算子本身容易和论文中的单序列伪代码一一对应。
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

