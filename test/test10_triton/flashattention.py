import torch
import triton

import triton.language as tl

from my_vllm.attention.triton_flash_attention import flash_attention_v1


def test_flash_attention_v1_matches_torch_prefill():
    if not torch.cuda.is_available():
        return
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


def test_flash_attention_v1_matches_torch_decode():
    if not torch.cuda.is_available():
        return
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
