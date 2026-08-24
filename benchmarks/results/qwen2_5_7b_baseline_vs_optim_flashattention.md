# Qwen2.5-7B Baseline vs 融合 Paged FlashAttention

## 同口径结果

模型 Qwen2.5-7B-Instruct-1M，单卡 BF16，256 个固定请求，并发 16；每请求
1024 Prompt tokens + 128 Output tokens；`max_model_len=4096`、Token Budget=2048、
KV blocks=4097、Prefix Cache 关闭。请求 SHA-256 为
`991592d2a65f7d456adb9909f7ead6a272095d8da4dc14c518a5622e08ae04bc`。

| 指标 | Gather-based Baseline | 融合 Paged FlashAttention | 变化 |
|---|---:|---:|---:|
| 256 请求总耗时 | 182.733 s | 91.251 s | **-50.06%** |
| Request Throughput | 1.401 req/s | 2.805 req/s | **+100.25%** |
| Prompt Throughput | 1,434.576 tok/s | 2,872.785 tok/s | **+100.25%** |
| Output Throughput | 179.322 tok/s | 359.098 tok/s | **+100.25%** |
| Total Token Throughput | 1,613.898 tok/s | 3,231.884 tok/s | **+100.25%** |
| TTFT Mean | 310.87 ms | 313.75 ms | +0.93% |
| TTFT P99 | 1,809.35 ms | 1,680.00 ms | -7.15% |
| TPOT Mean | 87.30 ms | 42.33 ms | **-51.51%** |
| TPOT P99 | 88.45 ms | 45.02 ms | **-49.10%** |
| Client E2E Mean | 11,399.58 ms | 5,690.82 ms | **-50.08%** |
| GPU 显存 Peak | 19,484 MiB | 19,478 MiB | -6 MiB |
| GPU 利用率 Mean | 73.48% | 84.92% | +11.44 pct |
| 平均功耗 | 229.91 W | 341.53 W | +48.55% |
| 抢占次数 | 0 | 0 | 不变 |

## 结论

融合版本把 `block_table` 页表翻译、变长请求边界、分页 KV 二维加载、GQA head
映射、因果掩码和 online softmax 放进同一次 Triton Attention launch，消除了
Baseline 的 Python 逐请求循环、完整历史 KV gather、`repeat_interleave` 和多次
kernel launch。在不增加 KV 容量或进程峰值显存的情况下，总吞吐提升约 2.00 倍，
TPOT 减少约 51.5%。

TTFT Mean 没有改善，说明下一阶段 CUDA Graph 与调度/launch 路径优化仍有明确空间。
本结果是融合优化的整体效果；单个优化点的独立贡献需要消融实验验证。

详细配置、显存拆分、分位数与复现命令见
`qwen2_5_7b_optim_flashattention_256req_c16.md`。
