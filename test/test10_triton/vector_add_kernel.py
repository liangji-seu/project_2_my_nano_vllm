
'''
下面来理解一下五个概念：
    pid
        pid = program ID, 当前块的唯一标识符，代表该block,在整个网格grid中的位置是第几个块
        在一维网格中，可以用pid = tl.program_id(0)来获取这个blockIdx.x

    block_start
        当前块在全局数据中的起始位置，用于确保各块处理的数据范围互不重叠，且完整覆盖整个数据集。
        block_start = pid * BLOCK_SIZE

    offsets
        表示当前块内每个thread相对于块起始位置的偏移量，用于帮助每个线程计算其在全局数据汇总的具体索引

    mask
        创建掩码，防止线程访问超出数据范围之外的元素


    idx
        每个thread在全局数据中的具体索引，用于加载和存储数据。确保每一个线程处理唯一的数据元素。

'''

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):

    '''
    这个函数，应该就是可以理解为核函数一样
    '''
    # 1. 定义每个线程在全局数据中的具体索引
    pid = tl.program_id(0)
    block_start = pid* BLOCK_SIZE
    offsets = tl.arange(0,  BLOCK_SIZE)
    idx = block_start + offsets # BLOCK_SIZE个thread在全局数据中的起始坐标
    mask = idx < N # 启动的thread的id掩码

    # 2. 加载数据，并执行内核算法vec_add
    x = tl.load(X_ptr + idx, mask=mask)
    y = tl.load(Y_ptr + idx, mask=mask)

    # 3. 执行向量加法
    z = x + y

    # 4. 存储结果
    tl.store(Z_ptr  + idx, z, mask=mask)



def vector_add(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    # python wrapper for triton kernel
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    N = x.numel()
    z = torch.empty_like(x)

    # 划分每个block的任务
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](x,y,z,N,BLOCK_SIZE=BLOCK_SIZE)

    return z


if __name__ == "__main__":
    # 造两个随机向量，验证 triton kernel 结果 vs torch 原生计算
    torch.manual_seed(0)
    N = 4096  # 故意不用 1024 的整数倍，测试 mask 边界
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")

    z = vector_add(x, y)
    z_ref = x + y

    assert torch.equal(z, z_ref), f"结果不一致! max_err={torch.max(torch.abs(z - z_ref))}"
    print(f"✓ vector_add 结果与 torch 完全一致 (N={N})")

    # 顺手用 triton 自带的 benchmark 工具测一下耗时
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(12, 25)],  # 4096 ~ 16M
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="vector_add_perf",
            args={},
        )
    )
    def benchmark(N, provider):
        xx = torch.randn(N, device="cuda")
        yy = torch.randn(N, device="cuda")
        if provider == "triton":
            fn = lambda: vector_add(xx, yy)
        else:
            fn = lambda: xx + yy
        return triton.testing.do_bench(fn)  # 返回耗时(ms)

    benchmark.run(print_data=True, show_plots=False)
