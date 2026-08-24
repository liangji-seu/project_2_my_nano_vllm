# Qwen2.5-7B 三阶段性能对比

固定 workload 均为256请求、1024 prompt tokens、128 output tokens、并发16。

| 阶段 | Total Token Throughput | 平均 TPOT | 总耗时 | 峰值显存 |
|---|---:|---:|---:|---:|
| Gather Attention Baseline | 1,613.898 tok/s | 87.30 ms | 182.733 s | 19,484 MiB |
| 融合 Paged FlashAttention | 3,231.884 tok/s | 42.33 ms | 91.251 s | 19,478 MiB |
| Paged FlashAttention + FULL CUDA Graph | 3,269.680 tok/s | 41.77 ms | 90.196 s | 21,264 MiB |

从 Baseline 到最终版本：

- 总 Token Throughput 提升约 102.60%；
- 平均 TPOT 降低约 52.16%；
- 总耗时降低约 50.64%。

其中主要收益来自融合 Paged FlashAttention；CUDA Graph 在同机 Eager A/B 中进一步
提升吞吐 1.84%、降低 TPOT 1.94%，代价是约 1,786 MiB 的进程峰值显存。由于
FlashAttention 历史正式结果与 CUDA Graph 结果不在同一天，CUDA Graph 的正式
增益应引用同机对照 `3,210.711 → 3,269.680 tok/s`，而不是直接用历史值计算。

