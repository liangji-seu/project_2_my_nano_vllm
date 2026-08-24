# Qwen2.5-7B 纯 Decode FULL CUDA Graph 性能实验

## 1. 结论

在融合 PagedAttention、变长 Batch、Decode 2D 分页加载和 GQA 的
FlashAttention 版本上，增加启动阶段主动捕获的纯 Decode FULL CUDA Graph。
与同一天、同代码、同机器上的 `--disable-cuda-graph` 对照相比：

| 指标 | Eager 对照 | FULL CUDA Graph | 变化 |
|---|---:|---:|---:|
| 总耗时 | 91.853 s | 90.196 s | -1.80% |
| Request Throughput | 2.787 req/s | 2.838 req/s | +1.84% |
| Total Token Throughput | 3,210.711 tok/s | 3,269.680 tok/s | +1.84% |
| 平均 TTFT | 317.96 ms | 319.29 ms | +0.42% |
| 平均 TPOT | 42.59 ms | 41.77 ms | -1.94% |
| 平均 Client E2E | 5,728.44 ms | 5,624.99 ms | -1.81% |
| 抢占次数 | 0 | 0 | 不变 |

CUDA Graph 对 TTFT 没有收益，因为 Prefill、输入整理、调度和采样都在图外；收益
集中在重复执行的 Decode 模型 forward。Qwen2.5-7B 的单步计算量较大，CPU launch
开销占比有限，因此本实验的端到端增益约为 2%，不能描述成数量级优化。

## 2. 实验配置

| 项目 | 配置 |
|---|---|
| 日期 | 2026-08-24 |
| GPU | NVIDIA GeForce RTX 4090，48,499 MiB 可见显存，SM 8.9 |
| Driver | 580.159.03 |
| PyTorch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| Triton | 3.6.0 |
| 模型 | Qwen2.5-7B-Instruct-1M，BF16，TP=1，PP=1 |
| 分支 | `cudagraph/profile/qwen2.5_7B` |
| CUDA Graph 运行代码 | `b2beaa5` |
| `max_model_len` | 4096 |
| `max_num_seqs` | 16 |
| Token Budget | 2048 |
| KV block size / blocks | 16 tokens / 4097（含 null block） |
| Prefix Cache | 关闭 |
| CUDA Graph | 纯 Decode FULL，启动主动捕获 |
| Batch 档位 | 1、2、4、8、16 |
| Sequence 档位 | 256 到 4096，固定步长 256 |
| 捕获图数量 | 5 × 16 = 80 |

## 3. 固定工作负载

请求文件：
`benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl`

SHA-256：
`991592d2a65f7d456adb9909f7ead6a272095d8da4dc14c518a5622e08ae04bc`

| 项目 | 数值 |
|---|---:|
| 请求总数 | 256 |
| Closed-loop 并发 | 16 |
| Warmup | 2 requests |
| 每请求 Prompt | 1024 tokens |
| 每请求输出 | 128 tokens，`ignore_eos=true` |
| 每请求完整序列 | 1152 tokens |
| Prompt Token 总量 | 262,144 |
| Output Token 总量 | 32,768 |
| 总 Token 量 | 294,912 |

该请求文件与 Baseline、FlashAttention 实验完全相同。

## 4. CUDA Graph 实现边界

```text
GPUModelRunner.__init__
    └── 空 CUDAGraphDispatcher

initialize_kv_cache
    └── 初始化合法 (FULL, BatchDescriptor) 库

capture_model
    └── 遍历 80 个描述符
          ├── _dummy_run(NONE)：eager warmup
          └── _dummy_run(FULL)：Wrapper 主动捕获

execute_model
    ├── 纯 Decode 且精确命中 batch 档位：replay
    └── Prefill/混合 batch/未命中档位：eager
```

图内范围为 `embedding → 28层 Transformer → final norm`。`lm_head`、greedy
sampler、D2H、InputBatch/metadata 构造和 Scheduler 都在图外。当前未实现请求数
padding，因此只有请求数精确命中 1/2/4/8/16 时使用图。

## 5. Sequence bucket 消融与修正

最初使用 `256/512/1024/2048/4096` 的指数档位。该 workload 在 Decode 阶段的
长度为 1025–1152，会向上匹配到 2048，使 Paged FlashAttention 多扫描接近一倍
的 KV 上界。第一次测量结果为：

| 指标 | 指数 bucket CUDA Graph | 同机 Eager |
|---|---:|---:|
| Total Token Throughput | 3,186.302 tok/s | 3,210.711 tok/s |
| 平均 TPOT | 42.96 ms | 42.59 ms |

即吞吐下降 0.76%。因此正式版本将 sequence bucket 改为固定 256 步长，实际
Decode 映射到 1280 档位。修正后吞吐提升 1.84%。这说明 CUDA Graph 的静态
bucket 必须同时权衡图数量和 Attention 无效计算，不能只减少图数量。

## 6. 显存成本

启动 profiling 和 KV Cache 与之前实验一致：

| 指标 | 字节 | GiB |
|---|---:|---:|
| 模型权重 | 15,276,321,792 | 14.227 |
| Profiling 激活峰值增量 | 286,425,088 | 0.267 |
| 实际 KV Cache | 3,759,013,888 | 3.501 |
| KV 初始化后 PyTorch allocated | 19,045,016,576 | 17.737 |
| 捕获80张图后 PyTorch allocated | 19,354,757,120 | 18.026 |
| CUDA Graph allocated 增量 | 309,740,544 | 0.288 |
| KV 初始化后 PyTorch reserved | 19,258,146,816 | 17.936 |
| 捕图后 PyTorch reserved | 20,065,550,336 | 18.688 |
| CUDA Graph reserved 增量 | 807,403,520 | 0.752 |

`nvidia-smi` 的运行期进程峰值从 Eager 的 19,478 MiB 增至 21,264 MiB，增加
1,786 MiB。该口径还包含 CUDA driver、graph exec 和 PyTorch allocator 外的
显存，因此大于 `torch.cuda.memory_allocated()` 的增量。

## 7. 完整延迟和 GPU 指标

| 指标 | Eager | CUDA Graph |
|---|---:|---:|
| TTFT P50 / P95 / P99 | 229.85 / 564.92 / 1693.35 ms | 229.12 / 576.63 / 1693.00 ms |
| TPOT P50 / P95 / P99 | 43.03 / 45.49 / 45.51 ms | 42.21 / 44.57 / 44.66 ms |
| Client E2E P50 / P95 / P99 | 5774.83 / 6053.21 / 6809.76 ms | 5659.02 / 5967.74 / 6696.38 ms |
| GPU 利用率 Mean | 86.07% | 85.62% |
| GPU 功耗 Mean | 343.50 W | 350.01 W |
| GPU 功耗 Peak | 456.15 W | 453.98 W |

## 8. 复现命令

CUDA Graph：

```bash
CUDA_VISIBLE_DEVICES=0 my_vllm serve \
  --model /home/liangji/huggingface/Qwen2.5-7B-Instruct-1M \
  --host 127.0.0.1 --port 13311 \
  --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --num-gpu-blocks 4097 \
  --disable-prefix-caching
```

同机 Eager 对照只增加：

```text
--disable-cuda-graph
```

压测：

```bash
python benchmarks/benchmark_online_serving.py \
  --workload benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl \
  --concurrency 16 --warmup-requests 2 --gpu 0 \
  --output benchmarks/results/qwen2_5_7b_cudagraph_256req_c16.json
```

## 9. 结果边界

- 这是单机单卡、固定并发16的一轮同机 A/B，不是多轮置信区间。
- FULL 图只覆盖纯 Decode；TTFT 不应预期明显下降。
- 当前没有请求数 padding，batch size 变化会使一部分 step 回退 eager。
- 80 张图用显存换取更细 sequence bucket；后续可研究动态 kernel 参数或更优档位。
- 相比历史 FlashAttention 正式结果 3,231.884 tok/s，本轮 CUDA Graph 为
  3,269.680 tok/s（+1.17%），但跨日期数据只作参考，正式收益采用同机 Eager A/B。

