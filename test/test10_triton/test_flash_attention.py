import pytest

torch = pytest.importorskip("torch")

from my_vllm.attention.triton_flash_attention import (
    flash_attention_v1,
    paged_varlen_flash_attention_v1,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_flash_attention_v1_matches_torch_prefill():
    torch.manual_seed(1)
    q = torch.randn((37, 4, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual = flash_attention_v1(q, k, v, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        is_causal=True,
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_flash_attention_v1_matches_torch_decode():
    torch.manual_seed(2)
    q = torch.randn((1, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((73, 4, 64), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    actual = flash_attention_v1(q, k, v, causal=True, causal_offset=72)
    # decode 的唯一 query 可以看到全部 73 个 K/V。
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _make_paged_cache(sequences, block_tables, block_size, num_blocks):
    """把连续 K/V 测试数据写入乱序物理页，便于验证页表翻译。"""

    device = sequences[0][0].device
    dtype = sequences[0][0].dtype
    num_kv_heads = sequences[0][0].shape[1]
    head_dim = sequences[0][0].shape[2]
    cache = torch.zeros(
        (num_blocks, 2, block_size, num_kv_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    for request_id, (key, value) in enumerate(sequences):
        for token_id in range(key.shape[0]):
            physical_block = block_tables[request_id][token_id // block_size]
            block_offset = token_id % block_size
            cache[physical_block, 0, block_offset] = key[token_id]
            cache[physical_block, 1, block_offset] = value[token_id]
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_paged_varlen_flash_attention_matches_torch_mixed_batch_gqa():
    """一次 launch 同时覆盖 chunked prefill、decode、分页读取和 GQA。"""

    torch.manual_seed(3)
    device = "cuda"
    dtype = torch.float16
    block_size = 16
    q_heads = 4
    kv_heads = 2
    head_dim = 64

    # req0: context=2 + query=3；req1: context=32 + decode query=1。
    query_lens = [3, 1]
    seq_lens_list = [5, 33]
    queries = [
        torch.randn((length, q_heads, head_dim), device=device, dtype=dtype)
        for length in query_lens
    ]
    sequences = [
        (
            torch.randn((length, kv_heads, head_dim), device=device, dtype=dtype),
            torch.randn((length, kv_heads, head_dim), device=device, dtype=dtype),
        )
        for length in seq_lens_list
    ]
    # block 0 留空，两个请求使用不连续、交错的物理页。
    block_rows = [[5], [3, 7, 2]]
    block_table = torch.zeros((2, 3), device=device, dtype=torch.int32)
    for request_id, row in enumerate(block_rows):
        block_table[request_id, : len(row)] = torch.tensor(
            row, device=device, dtype=torch.int32
        )
    cache = _make_paged_cache(sequences, block_rows, block_size, num_blocks=8)
    query = torch.cat(queries)
    query_start_loc = torch.tensor([0, 3, 4], device=device, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)

    actual = paged_varlen_flash_attention_v1(
        query,
        cache,
        block_table,
        query_start_loc,
        seq_lens,
        block_size=block_size,
        max_query_len=3,
        max_seq_len=33,
        causal=True,
    )

    expected_parts = []
    repeats = q_heads // kv_heads
    for q, (key, value), seq_len in zip(queries, sequences, seq_lens_list):
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        q_pos = seq_len - q.shape[0] + torch.arange(q.shape[0], device=device)
        kv_pos = torch.arange(seq_len, device=device)
        mask = kv_pos[None, :] <= q_pos[:, None]
        expected_parts.append(
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
                attn_mask=mask,
            ).transpose(0, 1)
        )
    expected = torch.cat(expected_parts)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
