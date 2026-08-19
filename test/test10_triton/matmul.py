import torch
import triton
import triton.language as tl


# 实现一个triton的matmul内核，一个block可以处理(128,64) x (64,128)的矩阵
@triton.jit
def _fused_linear_kernel_fwd(
        x_ptr, # 输入矩阵首元素指针
        w_ptr, # 权重矩阵首元素指针
        z_ptr, # 输出结果地址
        M,N,K, # 矩阵维度
        BLOCK_SIZE_M : tl.constexpr = 128, # 块大小
        BLOCK_SIZE_N : tl.constexpr = 128,
        BLOCK_SIZE_K : tl.constexpr = 64
):
    # 获取该block在grid内的编号
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 开始划分我们一个block的处理数据范围：
    # offs_m 为 该block负责的A的行索引范围：A_sub
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]#(BLOCK_SIZE_M, 1) 的列向量
    # offs_n 为 该block负责的B的列索引范围：B_sub
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]#(1，BLOCK_SIZE_N) 的行向量

    # 创建C_sub的缓冲区，共享内存？
    '''
    这里既不是共享显存，也不是全局显存。

    这个仅仅只是一个逻辑张量，和硬件里的物理分布之间，隔了一层编译器管理的layout。

  z = tl.zeros((128, 128)) 在 Triton 里是一个纯值（SSA 
  值）——它没有地址，你永远拿不到它的指针。Triton 的内存模型只有三层，z 属于中间那层的"寄存器"：

  ┌────────────────────┬─────────────────────────────┬───────────────────────────────┐
  │         层         │       你能否显式控制        │            谁在管             │
  ├────────────────────┼─────────────────────────────┼───────────────────────────────┤
  │ 全局显存（DDR/L2） │ tl.load / tl.store 显式读写 │ 你                            │
  ├────────────────────┼─────────────────────────────┼───────────────────────────────┤
  │ 共享显存（smem）   │ 不可见、不可控              │ 编译器（tl.dot 的操作数搬运） │
  ├────────────────────┼─────────────────────────────┼───────────────────────────────┤
  │ 寄存器             │ 不可见、不可控              │ 编译器（所有"值"的安放）      │
  └────────────────────┴─────────────────────────────┴───────────────────────────────┘


    在cuda官方的matmul算子的实现中，

    教材版 CUDA matmul 写的是 __shared__ float 
    C[BLOCK][BLOCK]——但那是慢版本。真实的高性能 CUDA
    kernel（cuBLAS/cutlass/FlashAttention）里，累加器都是寄存器分块（register tiling）：每线程持有
    BLOCK_M/(blockDim.y) × BLOCK_N/(blockDim.x) 个累加元素，smem 只用来中转 A/B 的 tile。Triton
    默认就是 register tiling

    所以，我们之前学习的cuda matmul，这个累加缓冲区本质上也是存放在thread的各自的寄存器里面的，共享内存只是用来
    保存A_tile, B_tile的缓冲区。累加在各自的thread寄存器里面。

    所以这里的z，我们编写triton的时候，不需要关系底层的硬件，编译器会自动划分硬件

    这里的z，就是单独的逻辑张量，没有地址，实际是他是经过中间层，底层被分散到各自的thread的寄存器当中了。
    
    '''
    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 开始进行共享内存+缓冲区
    for k in range(0, K, BLOCK_SIZE_K):
        # 每一个A_tile x B_tile

        # A_sub内的A_tile的行偏移 = B_sub内本轮的B_tile的列偏移
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k

        # x_ptr + offs_m = A_sub首地址
        # A_sub首地址 + x_k = A_tile首地址
        # 加一个边界情况的mask, offs_m, x_k不能移动到A的外面去
        x = tl.load(x_ptr + offs_m * K + x_k, mask=(offs_m < M) & (x_k < K), other=0.0)
        x = x.to(tl.float16)


        # B_sub内的B_tile的列偏移
        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k

        w = tl.load(w_ptr + w_k*N + offs_n, mask=(w_k < K)&(offs_n < N), other=0.0)
        w = w.to(tl.float16)

        # 分块相乘, 累加到z的共享内存缓冲区
        z = tl.dot(x, w, acc=z)

    # 输出矩阵的z的这个C_tile的首地址
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m<M) & (offs_n < N) # 这个block负责写入的块不能写到C的外面

    tl.store(z_ptr + z_offset, z, mask=z_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # python wrapper：把 A (M,K) × B (K,N) 切成 (BLOCK_SIZE_M, BLOCK_SIZE_N) 的 C 块，
    # 每个块一个 program（pid_m, pid_n）负责。
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    _, N = b.shape
    assert a.shape[1] == b.shape[0], "K 维必须相等"

    z = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))  # 每个块覆盖 128x128 的输出
    _fused_linear_kernel_fwd[grid](
        a, b, z, M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
    )
    return z


if __name__ == "__main__":
    torch.manual_seed(0)

    # 测试 1：你指定的场景 —— 一个 block 处理 (128,64) x (64,128)，grid=(1,1)
    M, K, N = 128, 64, 128
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    z = matmul(a, b)
    z_ref = (a.float() @ b.float())  # fp16 输入，fp32 参考结果
    err = torch.max(torch.abs(z - z_ref)).item()
    assert torch.allclose(z, z_ref, atol=1e-2, rtol=1e-2), f"结果不一致! max_err={err}"
    print(f"✓ 一个 block 完成 {M}x{K} @ {K}x{N}，max_err={err:.4f} (fp16 精度内)")

    # 测试 2：非整除的维度 —— 逼 mask 生效 (M=100,N=150 不能被 128 整除, K=80 不能整除 64)
    M, K, N = 100, 80, 150
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    z = matmul(a, b)
    z_ref = (a.float() @ b.float())
    err = torch.max(torch.abs(z - z_ref)).item()
    assert z.shape == (M, N) and torch.allclose(z, z_ref, atol=1e-2, rtol=1e-2), f"结果不一致! max_err={err}"
    print(f"✓ 非整除维度 {M}x{K} @ {K}x{N}，max_err={err:.4f} (mask 生效)")

    # 测试 3：多 block 场景 (grid=(2,2))
    M, K, N = 256, 128, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    z = matmul(a, b)
    z_ref = (a.float() @ b.float())
    err = torch.max(torch.abs(z - z_ref)).item()
    assert torch.allclose(z, z_ref, atol=1e-2, rtol=1e-2), f"结果不一致! max_err={err}"
    print(f"✓ 多 block {M}x{K} @ {K}x{N} (grid=(2,2))，max_err={err:.4f}")

    # 性能对比（别期待赢 cuBLAS —— naive 版本没有 swizzle 优化）
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["K"],
            x_vals=[256, 512, 1024, 2048],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="matmul_perf",
            args={"M": 1024, "N": 1024},
        )
    )
    def benchmark(M, N, K, provider):
        aa = torch.randn(M, K, device="cuda", dtype=torch.float16)
        bb = torch.randn(K, N, device="cuda", dtype=torch.float16)
        if provider == "triton":
            fn = lambda: matmul(aa, bb)
        else:
            fn = lambda: aa @ bb
        return triton.testing.do_bench(fn)  # 返回耗时(ms)

    benchmark.run(print_data=True, show_plots=False)


