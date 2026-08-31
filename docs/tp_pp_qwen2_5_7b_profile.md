# Qwen2.5-7B TP=2 / PP=2 异步流水线实验

非对称四卡拓扑的完整开发、失败实验与 PP>2 状态分叉排查过程，见 [非对称四卡环境下 TP/PP 开发与问题排查日志](tp_pp_asymmetric_4gpu_debug_log.md)。

## 实验目标

验证以下实现能够在单机四卡上正确运行，并记录可复现的性能基线：

- TP=2：QKV/gate-up 列并行，O/down 行并行，词表并行 embedding/lm_head。
- PP=2：28 层按全局 layer index 连续切为 14/14 层。
- ModelRunner V2：`execute_model()` 只执行前向，`sample_tokens()` 独立完成 logits 与采样。
- EngineCore 使用 `step_with_batch_queue`，维护深度为 `pipeline_parallel_size=2` 的在途 batch queue。
- PP 最后 stage 通过独立 CUDA stream 和 sibling NCCL communicator 广播采样 token，非最后 stage 延迟消费并校验 request/token 状态。

本实验关闭 CUDA Graph，避免把图优化收益混入 TP/PP 数据。

## 环境

- 日期：2026-08-31
- 模型：Qwen2.5-7B，BF16
- 模型结构：28 layers，hidden size 3584，28 Q heads，4 KV heads
- GPU：4 × NVIDIA GeForce RTX 4080 SUPER 32 GiB
- PyTorch：2.7.1+cu126
- CUDA：12.6
- NCCL：2.26.2
- GPU P2P：不支持；NCCL 经 PCIe/主机路径通信
- 代码分支：`TP_PP/profile/qwen2.5_7B`
- 异步队列实现基点：`9d47532`

新服务器的拓扑不是对称双 NUMA：GPU0 位于 NUMA0，GPU1/2/3 位于 NUMA1；GPU1↔GPU2 为 PIX。四卡不能组成两组都在各自 NUMA 内的 TP=2。

最终采用的逻辑 rank 到物理卡映射为：

| PP stage | 逻辑 ranks | 物理 GPUs | TP 链路 |
|---|---:|---:|---|
| stage 0（layers 0–13） | 0, 1 | 0, 3 | SYS，跨 NUMA |
| stage 1（layers 14–27 + norm/lm_head） | 2, 3 | 1, 2 | PIX，同 NUMA |

启动时使用 `CUDA_VISIBLE_DEVICES=0,3,1,2`，因此 PP 通信 lane 为物理 GPU0↔GPU1、GPU3↔GPU2。

## 固定压力用例

- requests：256
- concurrency：32
- 每条 prompt：1024 tokenizer tokens
- 每条最大输出：128 tokens
- `max_model_len=4096`
- `max_num_seqs=32`
- `max_num_batched_tokens=2048`
- KV block size：16
- `gpu_memory_utilization=0.82`
- prefix cache：关闭
- CUDA Graph：关闭
- warmup：4 requests，不计入正式计时
- 采样：greedy

prompt content 由 `(" hello" * 1021)` 构造；HTTP 层添加 `[user]: ` 后，经模型 `tokenizer.json` 验证恰好为 1024 tokens。关闭 prefix cache 是必要条件，否则相同 prompt 会命中 warmup 建立的前缀缓存，无法测到真实 prefill 成本。

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,3,1,2 my_vllm serve \
  --model /root/autodl-tmp/liangji \
  --host 0.0.0.0 --port 8000 \
  --tp-size 2 --pp-size 2 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.82 \
  --disable-prefix-caching \
  --disable-cuda-graph
```

压测命令：

```bash
python benchmarks/profile_tp_pp.py \
  --model /root/autodl-tmp/liangji \
  --repeat-word hello --repeat-count 1021 \
  --max-tokens 128 \
  --requests 256 --concurrency 32 --warmup 4 \
  --tokenizer /root/autodl-tmp/liangji/tokenizer.json
```

## 正确性验证

- 回归测试：24 passed。
- 强制队列测试：将 token budget 降至 4，以 8 个并发请求制造多个在途 micro-batch。
- 8/8 请求均生成完全相同的 8-token 输出。
- 服务日志无 NCCL 死锁、Worker/RPC 异常或 PP 延迟 token 状态分叉。

## 正式结果

| 指标 | 结果 |
|---|---:|
| 完成请求 | 256 / 256 |
| 总耗时 | 65.9614 s |
| 请求吞吐 | 3.88 req/s |
| 输出 token 数 | 32,768 |
| 输出 token 吞吐 | 496.78 tok/s |
| prompt + output 总 token 吞吐 | 4,470.98 tok/s |
| 平均 E2E latency | 8,221.96 ms |
| P50 E2E latency | 8,197.90 ms |
| P95 E2E latency | 9,533.01 ms |
| 每卡峰值显存 | 27,130–27,168 MiB |

`total token throughput = (256 × 1024 + 32,768) / 65.9614`。输出 token 数由 `tokenizer.json` 实际编码响应文本得到，不使用字符数代替 token 数。

当前 HTTP API 是非流式接口，只能在完整响应返回时打点。因此本实验不报告伪造的 TTFT/TPOT；后续接入逐 token streaming timestamp 后再测真实 TTFT、TPOT 和 ITL 分布。

## 显存与利用率

- 每个 rank 局部模型权重：约 3.553 GiB。
- 每个 rank 持有 14 个 attention layer。
- 每个物理 KV block：229,376 bytes/rank。
- 统一 KV block 数：101,348。
- 实际 KV tensor：约 21.650 GiB/rank。
- 正式区间每卡峰值显存：约 26.49–26.53 GiB（27,130–27,168 MiB）。

正式计时区间内的 GPU 采样（200 ms）：

| 物理 GPU | 所属 stage | 平均利用率 | 峰值利用率 | 峰值显存 |
|---:|---|---:|---:|---:|
| 0 | stage 0 | 96.34% | 100% | 27,130 MiB |
| 3 | stage 0 | 98.46% | 100% | 27,168 MiB |
| 1 | stage 1 | 71.46% | 100% | 27,152 MiB |
| 2 | stage 1 | 72.28% | 100% | 27,152 MiB |

stage 0 的跨 NUMA TP 链路成为流水线瓶颈，stage 1 存在明显 bubble。该现象属于本机非对称 PCIe/NUMA 拓扑的限制，不应被解释为 PP 已达到理想负载均衡。

## 物理卡映射对照

交换两个 stage 的物理卡映射（`CUDA_VISIBLE_DEVICES=1,2,0,3`）后，在完全相同配置下得到：

| 映射 | 总耗时 | 输出吞吐 | 平均 E2E | P95 E2E |
|---|---:|---:|---:|---:|
| `0,3,1,2` | 65.9614 s | 496.78 tok/s | 8,221.96 ms | 9,533.01 ms |
| `1,2,0,3` | 66.6008 s | 492.01 tok/s | 8,301.32 ms | 9,558.23 ms |

第一种映射的输出吞吐高约 0.97%，故保留为本机记录结果。该差异较小，不能据此泛化到具有 NVLink 或对称 NUMA 拓扑的服务器。

## TP=1 / PP=4 对照实验

保持模型、请求、并发、token budget、显存利用率、warmup、prefix cache 和 CUDA Graph 开关全部一致，只把并行配置改为 TP=1、PP=4。物理卡顺序为 `GPU0 → GPU1 → GPU2 → GPU3`，每个 stage 持有连续 7 层，异步 batch queue 深度为 4。

### PP 深度状态校验修复

首次正式 PP=4 warmup 暴露了延迟 token 校验只适用于 PP=2 的问题：旧逻辑假设旁路 token 被消费时，本地 output 最多只比接收时多一项，并检查列表最后一个 token。PP=4 延迟四步消费时，SchedulerOutput 可能已经同步回填该 token 及多个后续 token，因此旧 token 已不在列表末尾。

修复后使用接收时的 `expected_len` 作为稳定位置：若本地 output 已经更长，则校验 `output_token_ids[expected_len]` 是否等于延迟 token。新增 PP>2 专项回归测试后，完整测试集为 25 passed。失败的首次压测未计入任何性能指标。

### PP=4 显存布局

| stage / GPU | 局部层 | 局部权重 | 正式区间峰值显存 |
|---|---:|---:|---:|
| stage 0 / GPU0 | 7 + embedding | 4.064 GiB | 26,566 MiB |
| stage 1 / GPU1 | 7 | 3.049 GiB | 25,616 MiB |
| stage 2 / GPU2 | 7 | 3.049 GiB | 25,616 MiB |
| stage 3 / GPU3 | 7 + norm/lm_head | 4.064 GiB | 26,564 MiB |

- 每 rank KV block：229,376 bytes。
- 统一 KV block 数：98,102。
- stage 0/3 的 embedding 或 lm_head 权重更多，决定了统一 KV block 数的下限。

### PP=4 正式结果

| 指标 | TP=2 / PP=2 | TP=1 / PP=4 | 变化 |
|---|---:|---:|---:|
| 总耗时 | 65.9614 s | 60.0607 s | -8.95% |
| 请求吞吐 | 3.88 req/s | 4.26 req/s | +9.79% |
| 输出 token 吞吐 | 496.78 tok/s | 545.58 tok/s | +9.82% |
| prompt + output 总 token 吞吐 | 4,470.98 tok/s | 4,910.23 tok/s | +9.82% |
| 平均 E2E latency | 8,221.96 ms | 7,492.58 ms | -8.87% |
| P50 E2E latency | 8,197.90 ms | 7,481.29 ms | -8.74% |
| P95 E2E latency | 9,533.01 ms | 8,457.09 ms | -11.29% |

PP=4 正式计时区间的 GPU 采样：

| GPU / stage | 平均利用率 | 峰值利用率 | 峰值显存 |
|---|---:|---:|---:|
| GPU0 / stage 0 | 99.20% | 100% | 26,566 MiB |
| GPU1 / stage 1 | 99.75% | 100% | 25,616 MiB |
| GPU2 / stage 2 | 99.77% | 100% | 25,616 MiB |
| GPU3 / stage 3 | 77.17% | 100% | 26,564 MiB |

在这台 GPU P2P 不可用的服务器上，TP=2 的集合通信需要经过 PCIe/主机路径；PP=4 虽然增加了流水线 stage 数和 bubble，但只传递 activation，最终输出吞吐反而提高 9.82%。这是特定硬件拓扑下的结果；在 NVLink/NVSwitch 机器上，TP=2 的相对表现可能明显改善。
