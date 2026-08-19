

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE:tl.constexpr):
    row_idx = tl.program_id(0) # grid内的本block的Idx.x

    row_start_ptr = input_ptr + row_idx * input_row_stride # 本block处理的数据的起始地址

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets


    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))  # padding 位置填 -inf，不影响 max/sum

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)



def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    
    if BLOCK_SIZE >=4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](x,
            y,
            x.stride(0),
            y.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE)

    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    # 正确性验证：故意用非 2 的幂的 N，让 mask + other=-inf 真正生效
    M, N = 64, 1000
    x = torch.randn(M, N, device="cuda")
    z = softmax(x)
    z_ref = torch.softmax(x, dim=-1)

    err = torch.max(torch.abs(z - z_ref)).item()
    assert torch.allclose(z, z_ref, atol=1e-6, rtol=1e-5), f"结果不一致! max_err={err}"
    print(f"✓ softmax 结果与 torch 一致 (M={M}, N={N}, max_err={err:.2e})")

    # softmax 的基本性质：每行和为 1
    row_sums = z.sum(dim=-1)
    print(f"  每行和的最大偏差: {torch.max(torch.abs(row_sums - 1)).item():.2e}")

    # 性能对比
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(8, 13)] + [1000, 3000],  # 256~4096 + 非 2 幂
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="softmax_perf",
            args={"M": 512},
        )
    )
    def benchmark(N, M, provider):
        xx = torch.randn(M, N, device="cuda")
        if provider == "triton":
            fn = lambda: softmax(xx)
        else:
            fn = lambda: torch.softmax(xx, dim=-1)
        return triton.testing.do_bench(fn)  # 返回耗时(ms)

    benchmark.run(print_data=True, show_plots=False)
