# Qwen2.5-7B gather-based Attention Baseline

## 1. 测试目的

记录my-vLLM框架阶段的单卡性能基线。该分支保留旧Attention路径：当前token
通过`slot_mapping`写入分页KV Cache；执行Attention前，由Python逐请求根据
`block_table` gather完整历史KV，再通过`repeat_interleave`把4个KV heads复制
成28个heads，最后调用教学版Triton FlashAttention。

后续PagedAttention、变长Batch、原生GQA、CUDA Graph、TP/PP和投机解码均应
复用同一请求文件，并以本报告为对照。

## 2. 环境与配置

| 项目 | 配置 |
|---|---|
| 日期 | 2026-08-21 |
| GPU | NVIDIA GeForce RTX 4090，48,499 MiB可见显存，SM 8.9 |
| Driver | 580.159.03 |
| PyTorch / CUDA | 2.11.0+cu130 / CUDA 13.0 |
| Triton | 3.6.0 |
| 模型 | Qwen2.5-7B-Instruct-1M，BF16，TP=1，PP=1 |
| my-vLLM分支 | `baseline/profile/qwen2.5_7B` |
| Baseline起点 | `04b8051` |
| `max_model_len` | 4096 |
| `max_num_seqs` | 16 |
| `max_num_batched_tokens` | 2048 |
| KV block size | 16 tokens |
| KV blocks | 4097（含1个null block，可用4096个） |
| Prefix Cache | 关闭，避免跨请求重复前缀影响结果 |

## 3. 固定工作负载

请求文件：
`benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl`

SHA-256：
`991592d2a65f7d456adb9909f7ead6a272095d8da4dc14c518a5622e08ae04bc`

| 项目 | 数值 |
|---|---:|
| 请求总数 | 256 |
| 并发窗口 | 16 |
| 每请求Prompt | 1024 tokens（真实Qwen tokenizer逐条校验） |
| 每请求输出 | 固定128 tokens，`ignore_eos=true` |
| Prompt Token总量 | 262,144 |
| Output Token总量 | 32,768 |
| 总Token量 | 294,912 |

`max_model_len=4096`表示服务与KV页表按4096上下文配置；本轮每条实际序列长度
为1152。若要求256条请求同时常驻且全部增长至4096，仅KV就需要65,536个可用
block，超过这张单卡容量。因此这里采用256个总请求、16个并发常驻请求，符合
在线服务常用的closed-loop并发压测口径。

## 4. 显存结果

### 4.1 启动Profiling拆分

| 指标 | 字节 | GiB |
|---|---:|---:|
| GPU总显存 | 50,854,952,960 | 47.362 |
| `gpu_memory_utilization=0.9`预算 | 45,769,457,664 | 42.626 |
| 模型权重实际分配 | 15,276,321,792 | 14.227 |
| 模型加载后总已用（含CUDA/NCCL等） | 16,170,745,856 | 15.060 |
| Profiling forward激活峰值增量 | 286,425,088 | 0.267 |
| Profiling计算出的可用KV预算 | 29,312,286,720 | 27.299 |
| 本次实际分配KV Cache | 3,759,013,888 | 3.501 |
| KV初始化后PyTorch allocated | 19,045,016,576 | 17.737 |
| KV初始化后PyTorch reserved | 19,258,146,816 | 17.936 |

本次没有使用自动算出的27.299 GiB KV预算，而是按压测最大并发容量显式分配
4097 blocks。原因是Baseline会在运行时额外gather历史KV并把GQA KV heads复制
扩展；当前`profile_run()`没有覆盖这些临时张量。若把剩余显存全部提前交给KV
Cache，长上下文运行时存在OOM风险。

### 4.2 压测期间GPU进程总显存

`nvidia-smi`以100ms周期采样1825次：

| 指标 | MiB | GiB |
|---|---:|---:|
| 最小 | 19,262 | 18.811 |
| 平均 | 19,483.88 | 19.027 |
| 峰值 | 19,484 | 19.027 |

`nvidia-smi`是整个GPU进程口径，包含PyTorch张量、CUDA context、NCCL、Triton
编译缓存及临时workspace，因此不能与上表各项直接重复相加。

## 5. 在线推理性能

### 5.1 吞吐

| 指标 | 结果 |
|---|---:|
| 总耗时 | 182.733 s |
| Request Throughput | 1.401 req/s |
| Prompt Throughput | 1,434.576 tok/s |
| Output Throughput | 179.322 tok/s |
| Total Token Throughput | 1,613.898 tok/s |
| 抢占次数 | 0 |

Prompt Throughput包含Chunked Prefill阶段处理的输入token；Output Throughput用于
衡量Decode能力，后续Attention/CUDA Graph版本应重点与179.322 tok/s比较。

### 5.2 延迟分布

| 指标 | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | 310.87 | 225.38 | 498.20 | 569.60 | 1,809.35 | 2,028.16 |
| TPOT (ms) | 87.30 | 87.62 | 88.36 | 88.40 | 88.45 | 88.48 |
| Client E2E (ms) | 11,399.58 | 11,380.04 | 11,545.14 | 11,606.59 | 12,835.42 | 13,221.50 |
| Engine E2E (ms) | 11,398.32 | 11,379.11 | 11,544.11 | 11,604.71 | 12,832.79 | 13,218.41 |

TTFT从FastAPI收到请求开始，到EngineCore完成该请求第一个采样token结束，包含
HTTP解析、ZMQ发送、等待调度和Prefill。TPOT为首token之后的生成耗时除以其余
127个输出token。HTTP接口当前为非流式，因此TTFT由引擎内部时间戳测量。

## 6. GPU利用率与功耗

| 指标 | Mean | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| GPU利用率 (%) | 73.48 | 71 | 88 | 92 | 93 | 96 |
| 功耗 (W) | 229.91 | 187.76 | 375.02 | 398.33 | 422.80 | 446.76 |

平均GPU利用率只有73.48%，说明该Baseline仍存在明显的Python逐请求循环、KV
gather/GQA复制、多次Kernel Launch及同步调度间隙。后续融合PagedAttention、
变长Batch和CUDA Graph的目标，是提高GPU利用率、降低TPOT并提升输出吞吐。

## 7. 复现命令

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
  --output benchmarks/results/qwen2_5_7b_baseline_256req_c16.json
```

原始逐请求指标和全部GPU采样点保存在同名JSON文件中。
