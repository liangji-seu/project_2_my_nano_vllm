# Qwen2.5-7B 融合 Paged FlashAttention 优化实验

## 1. 实验目的与结论

本实验在相同 my-vLLM 框架、模型、调度配置和固定请求集下，只替换 Attention
执行路径，测量融合 PagedAttention、原生变长 Batch、Decode 2D 分页 KV 加载、
原生 GQA 与 online softmax 的 Triton FlashAttention V1 相对 gather-based
Attention Baseline 的收益。本版本尚未启用 CUDA Graph。

核心结论：总 Token Throughput 从 **1,613.898 tok/s** 提升到
**3,231.884 tok/s（+100.25%）**；平均 TPOT 从 **87.30 ms** 降到
**42.33 ms（-51.51%）**；256 请求总耗时从 **182.73 s** 降到
**91.25 s（-50.06%）**。GPU 进程峰值显存基本不变，并且全程无抢占。

## 2. 唯一算法变量

### 2.1 Baseline 路径

```text
slot_mapping 写当前 K/V
        ↓
Python 逐 request 循环
        ↓
block_table gather 全部历史 KV，形成连续临时张量
        ↓
repeat_interleave：4 KV heads 复制为 28 heads
        ↓
逐 request 调用教学版 Triton FlashAttention
```

### 2.2 优化路径

```text
slot_mapping 写当前 K/V
        ↓
一次 kernel launch：grid = (Q tile, Q head, request)
        ↓
query_start_loc 在 kernel 内定位变长 request 边界
        ↓
block_table 在 kernel 内将逻辑 token 翻译为物理 KV block
        ↓
BLOCK_N × BLOCK_D 二维地址矩阵直接加载分页 K/V tile
        ↓
query_head // q_heads_per_kv_head 直接映射 GQA KV head
        ↓
因果掩码 + online softmax 累加输出
```

各优化点在代码中保持独立注释：

- 【PagedAttention优化】：页表翻译合并到 Attention kernel，消除完整历史 KV gather。
- 【变长Batch优化】：用 `query_start_loc` 从扁平 Q 中识别每个请求的真实区间。
- 【Decode 2D向量化加载优化】：以 `BLOCK_N × BLOCK_D` 地址矩阵读取分页 K/V tile。
- 【GQA优化】：28 个 Q heads 直接映射到 4 个 KV heads，不再复制扩展 KV。
- 【Prefill因果掩码】：依据 `context_len + local_q_pos` 屏蔽未来 token。

因此性能提升不能解释成单独某一个点的收益，而是上述融合 Attention 路径相对
旧 gather-based 路径的整体收益。若要做逐项归因，需要后续增加消融实验。

## 3. 环境与固定配置

| 项目 | 配置 |
|---|---|
| 日期 | 2026-08-22 |
| GPU | NVIDIA GeForce RTX 4090，48,499 MiB 可见显存，SM 8.9 |
| Driver | 580.159.03 |
| PyTorch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| Triton | 3.6.0 |
| 模型 | Qwen2.5-7B-Instruct-1M，BF16，TP=1，PP=1 |
| 分支 | `optim_flashattention/profile/qwen2.5_7B` |
| Attention 功能起点 | `5a63f45` |
| 正式压测运行代码 | `b09e967` |
| `max_model_len` | 4096 |
| `max_num_seqs` | 16 |
| `max_num_batched_tokens` / Token Budget | 2048 |
| KV block size | 16 tokens |
| KV blocks | 4097（1 个 null block，4096 个可用） |
| Prefix Cache | 关闭 |
| CUDA Graph | 关闭 |

模型结构为 28 层、hidden size 3584、28 个 Q heads、4 个 KV heads、head dim
128、intermediate size 18,944。调度仍为同步
`SchedulerOutput → execute_model → 等待结果`，并统一支持 Continuous Batching
与 Chunked Prefill。

## 4. 固定工作负载

请求文件：
`benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl`

SHA-256：
`991592d2a65f7d456adb9909f7ead6a272095d8da4dc14c518a5622e08ae04bc`

| 项目 | 数值 |
|---|---:|
| 请求总数 | 256 |
| Closed-loop 并发窗口 | 16 |
| Warmup | 2 requests |
| 每请求 Prompt | 1024 tokens（Qwen tokenizer 逐条校验） |
| 每请求输出 | 固定 128 tokens，`ignore_eos=true` |
| 每请求实际序列长度 | 1152 tokens |
| Prompt Token 总量 | 262,144 |
| Output Token 总量 | 32,768 |
| 总 Token 量 | 294,912 |

本实验和 Baseline 使用同一个已提交请求文件，而不是重新随机生成请求。

## 5. 显存 Profiling 与 KV Cache

启动阶段的 `profile_run()` 使用 2048 个 dummy tokens、采样 16 个 logit
positions。因为这条 profiling forward 不带运行时 Attention metadata，仍走连续
Q/K/V 的 profiling 路径，所以其模型权重与激活峰值测量和 Baseline 相同：

```text
requested_memory = total_memory × 0.9
available_kv     = requested_memory
                   - used_after_model_load
                   - activation_peak

                 = 45,769,457,664
                   - 16,170,745,856
                   - 286,425,088
                 = 29,312,286,720 bytes
```

| 指标 | 字节 | GiB |
|---|---:|---:|
| GPU 总显存 | 50,854,952,960 | 47.362 |
| 90% 显存预算 | 45,769,457,664 | 42.626 |
| 模型权重实际分配 | 15,276,321,792 | 14.227 |
| 模型加载后总已用 | 16,170,745,856 | 15.060 |
| Profiling forward 激活峰值增量 | 286,425,088 | 0.267 |
| Profiling 计算出的可用 KV 预算 | 29,312,286,720 | 27.299 |
| 本次实际分配 KV Cache | 3,759,013,888 | 3.501 |
| KV 初始化后 PyTorch allocated | 19,045,016,576 | 17.737 |
| KV 初始化后 PyTorch reserved | 19,258,146,816 | 17.936 |

每个 token 的全模型 KV 开销为 57,344 bytes；每个 16-token block 为
917,504 bytes。自动预算理论可分配 31,947 blocks，本实验固定使用 4097 blocks，
以确保与 Baseline 的 KV 容量完全一致。

压测期间 `nvidia-smi` 每 100 ms 采样，共 912 个样本：

| GPU 进程显存 | Baseline | 优化版 | 差值 |
|---|---:|---:|---:|
| Mean | 19,483.88 MiB | 19,478.00 MiB | -5.88 MiB |
| Peak | 19,484.00 MiB | 19,478.00 MiB | -6.00 MiB |

旧路径的 gather/GQA 扩展张量是运行时临时分配并复用的显存；优化版消除了这类
逻辑 KV 副本，但固定 KV Cache、模型权重和 CUDA context 占据主要部分，所以
进程峰值显存仅下降 6 MiB。该数据说明本实验的吞吐收益并非通过增加显存容量获得。

## 6. 性能结果与 Baseline 对比

### 6.1 吞吐与总耗时

| 指标 | Baseline | 优化版 | 变化 |
|---|---:|---:|---:|
| 总耗时 | 182.733 s | 91.251 s | -50.06% |
| Request Throughput | 1.401 req/s | 2.805 req/s | +100.25% |
| Prompt Throughput | 1,434.576 tok/s | 2,872.785 tok/s | +100.25% |
| Output Throughput | 179.322 tok/s | 359.098 tok/s | +100.25% |
| Total Token Throughput | 1,613.898 tok/s | 3,231.884 tok/s | +100.25% |
| 抢占次数 | 0 | 0 | 不变 |

由于每条请求的输入和输出 token 数完全固定，各种吞吐指标的提升比例相同。

### 6.2 延迟分布

| 指标 | Baseline Mean | 优化 Mean | Mean 变化 | Baseline P99 | 优化 P99 |
|---|---:|---:|---:|---:|---:|
| TTFT | 310.87 ms | 313.75 ms | +0.93% | 1,809.35 ms | 1,680.00 ms |
| TPOT | 87.30 ms | 42.33 ms | -51.51% | 88.45 ms | 45.02 ms |
| Client E2E | 11,399.58 ms | 5,690.82 ms | -50.08% | 12,835.42 ms | 6,781.45 ms |
| Engine E2E | 11,398.32 ms | 5,689.62 ms | -50.08% | 12,832.79 ms | 6,777.88 ms |

更完整的优化版分位数：

| 指标 | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 313.75 | 225.40 | 521.69 | 550.73 | 1,680.00 | 1,853.72 |
| TPOT (ms) | 42.33 | 42.82 | 44.95 | 44.99 | 45.02 | 45.06 |
| Client E2E (ms) | 5,690.82 | 5,747.57 | 5,891.83 | 6,002.98 | 6,781.45 | 7,182.79 |
| Engine E2E (ms) | 5,689.62 | 5,746.41 | 5,890.92 | 6,001.53 | 6,777.88 | 7,178.99 |

TTFT Mean 基本持平且有 0.93% 波动，说明当前优化主要降低 Attention 密集的持续
Decode 成本；TTFT 仍包含 HTTP、ZMQ、同步调度等待、Chunked Prefill 和首次 Triton
执行等开销。不能把此次结果描述为 TTFT 全面降低。

### 6.3 GPU 利用率与功耗

| 指标 | Baseline | 优化版 | 变化 |
|---|---:|---:|---:|
| GPU 利用率 Mean | 73.48% | 84.92% | +11.44 个百分点 |
| GPU 利用率 P50 | 71% | 81% | +10 个百分点 |
| GPU 利用率 P95 | 92% | 100% | +8 个百分点 |
| 平均功耗 | 229.91 W | 341.53 W | +48.55% |
| 峰值功耗 | 446.76 W | 454.82 W | +1.80% |

融合 kernel 消除了 Python 逐请求循环、完整 KV gather、GQA 复制和多次 launch，
因此 GPU 更持续地工作，利用率与平均功耗同时上升。功耗上升不是隐藏成本：本轮
耗时减半，若用“平均功耗 × 时间”粗略估算 GPU 能耗，Baseline 约 42.01 kJ，
优化版约 31.17 kJ，约下降 25.8%；这只是基于采样均值的近似值，不等同于专业
功率计读数。

## 7. 结果边界

- 这是单机单卡、固定 closed-loop 并发 16 的一次正式测量，不是多轮置信区间。
- 优化版包含 PagedAttention、变长 Batch、2D KV 加载和 GQA 的组合收益，未做消融。
- CUDA Graph、TP/PP、投机解码、Prefix Cache 均未启用。
- 服务仍为同步调度，HTTP 接口非流式；TTFT 来自引擎内部时间戳。
- 本轮序列实际长度为 1152，不能外推为超长上下文性能。

## 8. 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 my_vllm serve \
  --model /home/liangji/huggingface/Qwen2.5-7B-Instruct-1M \
  --host 127.0.0.1 --port 13311 \
  --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --num-gpu-blocks 4097 \
  --disable-prefix-caching

python benchmarks/benchmark_online_serving.py \
  --workload benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl \
  --concurrency 16 --warmup-requests 2 --gpu 0 \
  --output benchmarks/results/qwen2_5_7b_optim_flashattention_256req_c16.json
```

## 9. 实验资产

| 资产 | 路径/版本 |
|---|---|
| Attention 功能起点 | `5a63f45` |
| 正式运行代码 | `b09e967` |
| 原始结果提交 | `f90621f` |
| 固定请求文件 | `benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl` |
| 原始逐请求与 GPU 采样 | `benchmarks/results/qwen2_5_7b_optim_flashattention_256req_c16.json` |
| 机器可读实验清单 | `benchmarks/results/qwen2_5_7b_optim_flashattention_256req_c16_manifest.json` |
| 独立对比摘要 | `benchmarks/results/qwen2_5_7b_baseline_vs_optim_flashattention.md` |
