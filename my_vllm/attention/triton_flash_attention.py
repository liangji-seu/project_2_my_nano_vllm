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
        q_ptr, # Q矩阵张量的起始地址
        kv_cache_ptr, # 该layer的kvcache 张量的起始地址
        block_table_ptr, # block_table张量的地址
        query_start_loc_ptr, # req划分q向量的一维张量的地址
        seq_lens_ptr, # 每个req序列长度的张量
        out_ptr, # 输出张量的起始地址
        max_seq_len, # 最长序列长度
        softmax_scale_log2, # softmax 缩放因子

        # Q矩阵[q_i, q_head, head_dim]，各个轴的步进元素个数
        stride_qm: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,

        # kvcache tensor [num_blocks, 2, block_size, kv_heads, head_dim]，各个轴的步进
        stride_cache_block: tl.constexpr,
        stride_cache_kv: tl.constexpr,
        stride_cache_token: tl.constexpr,
        stride_cache_head: tl.constexpr,
        stride_cache_d: tl.constexpr,

        # block table 的[num_reqs, max_blocks_per_req]，各个轴的步进
        stride_bt_req: tl.constexpr,
        stride_bt_block: tl.constexpr,

        # O矩阵[q_i, q_head, head_dim]，各个轴的步进元素个数
        stride_om: tl.constexpr, 
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,


        HEAD_DIM: tl.constexpr,# head_dim
        BLOCK_D: tl.constexpr, # 同head_dim, 必须是2的指数倍
        BLOCK_SIZE: tl.constexpr, # 每个block的token数
        BLOCK_M: tl.constexpr, # 每个program可以处理同req的32个q向量
        BLOCK_N: tl.constexpr, # 每个program每次循环处理 32个k 向量
        Q_HEADS_PER_KV_HEAD: tl.constexpr, # gqa, 多少个q头 共用一个kv头
        IS_CAUSAL: tl.constexpr,
        # num_warps、num_stages 是 Triton launch 的保留关键字，由launch机制
        # 识别和拦截，不作为kernel的显式形参。
    ):
        """分页 KV + 变长 batch 的 online-softmax 主循环。

        grid 三个维度分别表示 ``Q tile / Q head / request``。一个 program 只
        负责一个 request 的一个 Q head，因而不同请求之间天然隔离。
        """



        # 当前program， 确认自己在grid中领到的各自的任务：
        # 0. 处理这个req的第几个Q_tile
        # 1. 处理第几个q头
        # 2. 处理第几个req
        query_block = tl.program_id(0)
        query_head = tl.program_id(1)
        request_id = tl.program_id(2)

        # 【变长Batch优化】每个 program 通过 query_start_loc 找到自己请求在
        # 扁平 Q 中的边界。矩形 grid 中超过真实 query_len 的 tile 全程 mask。

        # 加载出当前req的起始token索引
        request_q_start = tl.load(query_start_loc_ptr + request_id) 

        # 加载出当前req的结束token的索引，也就是下一个req的起始索引
        request_q_end = tl.load(query_start_loc_ptr + request_id + 1)

        '''
        [request_q_start, request_q_end)
        '''

        # 计算出当前program实际需要计算的这个req的query有多少个向量
        query_len = request_q_end - request_q_start

        # 计算一下自己算这个query的哪一部分Q_tile，得到query_len里面的起点
        tile_q_start = query_block * BLOCK_M

        #提取这个req的整个序列长度
        seq_len = tl.load(seq_lens_ptr + request_id)
        context_len = seq_len - query_len #算一下这个req的kv上下文长度

        #获取本次要计算的q向量的索引张量[xx, xx+1, xx+2, xx+3,...xx+BLOCK_M-1](在这个seq的query内部的索引)
        local_q_pos = tile_q_start + tl.arange(0, BLOCK_M)

        #得到本次要计算的q向量的索引张量，在整个batch的Q矩阵里面的索引，也就是全局索引
        flat_q_pos = request_q_start + local_q_pos

        #算出每个q的head_dim的各自的元素的q内索引张量
        offs_d = tl.arange(0, BLOCK_D)

        #加载掩码，保证加载Q_tile的时候，不会超出真实的Q，用来处理边界情况
        q_mask = (local_q_pos[:, None] < query_len) & (
            offs_d[None, :] < HEAD_DIM
        )

        # 开始加载出Q_tile
        q = tl.load(
            q_ptr
            + flat_q_pos[:, None] * stride_qm
            + query_head * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0, #表示被mask掉的元素填什么值
        )


        #__________________________________________________________________________

        # 【GQA优化】Qwen2.5 的 cache 只保存真实 KV heads。多个 Q heads 通过
        # 整数映射直接共享一个 KV head，不再 repeat_interleave 复制 KV 张量。


        # 通过当前program负责的q头id，看看我们用哪个kv头号
        kv_head = query_head // Q_HEADS_PER_KV_HEAD
        offs_n = tl.arange(0, BLOCK_N) #先定义加载一个kv头的向量内部元素的相对索引

        # 先创建好缓冲区，m,用来保存每个q in Q_tile 的 max(xi)
        # l用来保存每个q下计算出的分母的局部和
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        # acc是用来累加未归一化的O_i的计算结果的，等循环结束后，统一用最新的l来归一化，然后写入O_i
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)





        '''
        query: 所有请求拼接后的 ``[total_query_tokens, q_heads, head_dim]``。

        kv_cache: ``[num_blocks, 2, block_size, kv_heads, head_dim]``。

        block_table: ``[num_reqs, max_blocks_per_req]`` 物理页表。

        query_start_loc: 请求在扁平 query 中的前缀和边界。

        seq_lens: 写入本轮 K/V 后各请求的完整序列长度。
        '''



        # 开始我们的外层循环，每次执行block_n个kv向量头
        for start_n in range(0, max_seq_len, BLOCK_N):
            # 计算当前block_n个kv向量头的各个k向量，在kv上下文中的索引
            key_pos = start_n + offs_n
            key_valid = key_pos < seq_len

            # 计算当前block_n个kv向量头，各自属于是req内的block索引
            logical_block = key_pos // BLOCK_SIZE
            # 以及各个kv向量头，在block内的偏移索引
            block_offset = key_pos % BLOCK_SIZE



            # 【PagedAttention优化】页表翻译被合并进 attention kernel：不再
            # gather 连续历史 KV。每个 key token 在使用时才解析物理 block。
            physical_block = tl.load(
                block_table_ptr # block_table的地址
                + request_id * stride_bt_req # 1. 先偏移到对应的req
                + logical_block * stride_bt_block, # 2. 各个kv头的block对应的地址列表
                mask=key_valid,
                other=0,
            )






            # block_n x block_d, 表示的是这个program需要用到的K_tile的内存空间

            # 【Decode 2D向量化加载优化】一次构造 BLOCK_N x BLOCK_D 地址矩阵，
            # 沿 token/head_dim 两维加载一个分页 K/V tile。K 用 [D,N] 供 q@k，
            # V 用 [N,D] 供 p@v；数据直接进入本 program 的片上工作集。
            k_mask = (offs_d[:, None] < HEAD_DIM) & key_valid[None, :]

            # k就是用2D向量方式，给load一个地址的2维张量，让他直接加载各自的元素即可。
            k = tl.load(
                kv_cache_ptr # kvcache tensor的起始地址
                + physical_block[None, :] * stride_cache_block # 加上得到 这些k头的各个物理块的起始地址
                + block_offset[None, :] * stride_cache_token # 加上各个k头在block内的各自的偏移
                + kv_head * stride_cache_head # 加上所在头的偏移量
                + offs_d[:, None] * stride_cache_d, #每个k头的地址开始，各自包括一个头的长度，因此加载出来是2D张量
                mask=k_mask,
                other=0.0,
            )


            # 先算出每一轮的局部打分
            scores = tl.dot(q, k) * softmax_scale_log2

            # [BLOCK_M(Q_tile中q数数), BLOCK_N(K_tile中k数)]
            # valid[i, j] 回答q_i 能不能attend k_tile中的第j个k_j
            '''
            这里的valid, 是实现数据有效性，屏蔽掉边界不足block_m的q(padding), 和不足block_n的k(padding)
            就是说，这个（block_m, block_n）的score，有哪些是有效的，不是算到padding上去了
            '''
            valid = (local_q_pos[:, None] < query_len) & key_valid[None, :]


            # 如果我们是因果的，所以valid还需要再加上因果掩码
            if IS_CAUSAL:
                # 【Prefill因果掩码】chunked prefill 的第一个 Q 对应 context_len；
                # decode 时 query_len=1，它自然可以看到 seq_len 内全部 KV。
                valid &= key_pos[None, :] <= ( # 这里就是保证valid一定是下三角
                    context_len + local_q_pos[:, None]
                )

            # 开始掩码生效，产生正确的score的局部打分
            scores = tl.where(valid, scores, -float("inf"))



            '''
                        k0   k1   k2   ... k31          axis=1 求 max
                q0   [  s00  s01  s02  ... s0,31 ]  →  max_j(s0j)   ┐
                q1   [  s10  s11  s12  ... s1,31 ]  →  max_j(s1j)   ├─→ 结果形状 [BLOCK_M]
                ...                                                      （每个 q 一个值）
                q31  [ s310  s311 ...          ]  →  max_j(s31j)  ┘
            '''
            # score=(block_m, block_n)， axis=1,表示把k向量维度消掉，就看q向量维度
            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max) # 更新最大值m_i

            alpha = tl.exp2(m_i - m_new) #算一下更新系数

            probabilities = tl.exp2(scores - m_new[:, None]) # 算一下本k_tile的局部分母
            l_new = l_i * alpha + tl.sum(probabilities, axis=1) # 更新旧分母，加上本次的分母



            v_mask = key_valid[:, None] & (offs_d[None, :] < HEAD_DIM)
            v = tl.load( # 向量化直接加载V_tile
                kv_cache_ptr
                + physical_block[:, None] * stride_cache_block
                + stride_cache_kv
                + block_offset[:, None] * stride_cache_token
                + kv_head * stride_cache_head
                + offs_d[None, :] * stride_cache_d,
                mask=v_mask,
                other=0.0,
            )

            # 计算未归一化的累加缓冲区，也就是O_tile
            acc = acc * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
            m_i = m_new
            l_i = l_new


        # 最终归一化
        output = acc / l_i[:, None]

        
        out_mask = (local_q_pos[:, None] < query_len) & (
            offs_d[None, :] < HEAD_DIM
        )

        # 向量化写入 O_tile
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
    max_query_len: int, # batch里最长的query数
    max_seq_len: int, # batch里req, 最长的完成序列kvcache长度
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

    query = query.contiguous() # 确保这个张量的底层内存是连续的
    output = torch.empty_like(query) # 每个注意力输出向量，都和q是一样的形状，所以和输入的Q矩阵也是一样的输出形状

    # triton kernel启动前的 分块tiling + 网格配置
    # 觉得把整个注意力计算切成多少个小块，每个小块交给那个GPU program 去算


    '''
    这边是定义一个program(block) 处理数据的规模spec
    '''
    # 一个block负责计算一个head，且必须是2的指数倍，所以对head_dim取2 power
    block_d = triton.next_power_of_2(head_dim) # q向量维度的切分，我们是以一个head来切分的，这里要求了head_dim必须是2的指数倍
    block_m = 32 # Q分块的大小，一个block负责计算32个q向量
    block_n = 32 # 一个block负责计算32个kv向量


    '''
    根据一个program的spec, 我们开始计算实需要多少个program， grid 就是 program的布局
    '''
    grid = (
        triton.cdiv(max_query_len, block_m), # 每block_m个q都需要一个program，每个req都有max_query_len个q，所以需要xxx个program
        query.shape[1], # 每个Q head必须有一个program，所以这个维度算出来的program所需的个数
        seq_lens.numel(),  # 每个req都需要单独一个program, 所以必须有req个数个program
    )


    _paged_varlen_flash_attention_v1_fwd_kernel[grid](
        query, # 整个batch的输入Q矩阵
        kv_cache, # 这一层的kvcache tensor
        block_table, # 整个batch的每个req对应的kv向量的block id列表
        query_start_loc, # 划分Q矩阵 req分布
        seq_lens, # 这个batch的各个req的完整序列长度的张量
        output, # 输出矩阵，和Q矩阵相同形状
        max_seq_len, # batch的req中，拥有的最长历史kv向量个数的个数，决定了flashattention要循环更新多少次
        # attention = softmax(QK^T / √d)。这里除了1/√d还乘log2(e)，
        # 因为kernel内使用exp2，需要把以e为底的指数换到以2为底的尺度。
        1.0 / math.sqrt(head_dim) * math.log2(math.e), 

        # torch.Tensor.stride()返回各轴移动一个元素所需的底层步进，*将元组
        # 解包成多个位置参数。Triton kernel收到的是裸指针，不知道Tensor的
        # 形状与内存布局，因此必须依靠这些stride自行计算元素地址。
        *query.stride(), # Q矩阵[q_i, q_head, head_dim], 每个轴的步进距离元组
        *kv_cache.stride(), # 该layer的kvcache 张量[num_blocks, 2, block_size, kv_heads, head_dim] 每个轴的步进距离元组
        *block_table.stride(), # [num_reqs, max_blocks_per_req]
        *output.stride(), # 同Q
        HEAD_DIM=head_dim,
        BLOCK_D=block_d, # 同head_dim, 必须是2的指数倍
        BLOCK_SIZE=block_size, # 一个block里面16个token
        BLOCK_M=block_m, # Q矩阵的q向量切分，一个program负责处理32个q向量
        BLOCK_N=block_n,  # kv矩阵切分，一个program负责处理32个kv向量
        Q_HEADS_PER_KV_HEAD=query.shape[1] // kv_cache.shape[3], # GQA的分组 q_head // kv_heads, 表示多少个q头共享一个kv头
        
        # IS_CAUSAL在kernel形参中由tl.constexpr标记，是编译期常量；因此
        # kernel内部的if IS_CAUSAL类似条件编译，不需要运行时分支。
        IS_CAUSAL=causal, # 因果，kernel本身是通用的，不会写死因果，比如可能会有双向注意力
        num_warps=4, # 每个program内部规定4 个warp

        # num_warps是每个program内并行的warp数量。num_stages是软件流水线
        # 深度：2表示双缓冲，计算第i个K/V tile时预取第i+1个tile，使加载与
        # 计算重叠；更大的值会增加预取深度，也会占用更多片上资源。
        num_stages=2,
    )
    return output
