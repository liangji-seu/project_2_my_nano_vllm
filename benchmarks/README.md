# my-vLLM HTTP 性能压测

`benchmark_matrix.py` 面向当前非流式 `/v1/chat/completions` 接口，扫描请求总数、最大并发数、输入 token 长度和最大输出预算。使用本地 `tokenizer.json`，不需要安装原版 vLLM，也不会自动下载数据或启动模型。

## 当前指标口径

- 输入长度包含服务端实际拼接的 `[user]: `，按 `add_special_tokens=True` 编码。
- E2E 延迟从客户端工作线程发出 HTTP 请求开始，到完整响应体读完为止；不含请求在客户端线程池中的等待。整批耗时包含限流排队及全部请求完成的时间。
- RPS = 成功请求数 / 整批耗时。失败请求会保留，失败率单独输出；有失败时不能作为正常性能结果。
- 当前 API 的 `usage` 实际是字符数，脚本不采用它。
- 输出 token 数通过返回文本重新编码估算，使用 `add_special_tokens=False`。原始生成 token 解码时会丢弃特殊 token，而且重新编码不保证恢复原 token 序列，因此输出 token 吞吐是**估算值**，不等于精确的模型生成 token 吞吐。
- `--output-lens` 是最大生成预算，EOS 可能导致提前结束。脚本报告实际返回长度，不以预算代替输出量。
- 非流式接口无法观测首 token 或逐 token 时间，TTFT、TPOT、ITL 均为 `null`。E2E / 输出长度也不能代替这些指标。
- 合成负载衡量受控长度下的性能；真实文本负载改善代表性。两者都不评估答案准确率。

## 先启动服务

在服务器第一个终端执行。下面是 TP=2；测试 PP=2 时，退出旧服务后改为 `--tp-size 1 --pp-size 2`，其他参数保持一致。

```bash
conda activate vllm_study
cd ~/AI_INFRA/projects/project_2_my_nano_vllm

export MODEL_DIR=/mnt/hdd/huggingface/Qwen2.5-7B-Instruct-1M
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1

my_vllm serve \
  --model "$MODEL_DIR" --served-model-name qwen2.5-7b \
  --dtype bfloat16 --host 0.0.0.0 --port 8000 \
  --tp-size 2 --pp-size 1 \
  --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8 \
  --disable-prefix-caching --disable-cuda-graph
```

使用已安装的 `my_vllm` 命令；当前 `cli.py` 没有模块执行入口，`python -m my_vllm.cli` 不会调用 `main()`。

## 合成负载：先小规模验证

另开终端，执行：

```bash
conda activate vllm_study
cd ~/AI_INFRA/projects/project_2_my_nano_vllm
export TOKENIZER=/mnt/hdd/huggingface/Qwen2.5-7B-Instruct-1M/tokenizer.json

python benchmarks/benchmark_matrix.py \
  --model qwen2.5-7b --tokenizer "$TOKENIZER" \
  --requests 4 --concurrency 2 --input-lens 128 --output-lens 16 \
  --warmup 1 --label tp2-smoke
```

再扫描两种请求总数、三种并发和三种输入长度：

```bash
python benchmarks/benchmark_matrix.py \
  --model qwen2.5-7b --tokenizer "$TOKENIZER" \
  --requests 32 128 --concurrency 1 4 8 \
  --input-lens 128 512 2048 --output-lens 128 \
  --warmup 2 --repeats 1 --seed 0 --label tp2
```

这组参数有 18 个配置组合、1,440 条正式请求，另加每个 case 的 warmup。加上 `--dry-run` 可在不发 HTTP 请求的情况下检查负载和预算。需要复测波动时使用 `--repeats 3`，请求量会相应增长。

合成请求带有不同的前缀，并验证最终输入 token 数达到指定值。长输入使用重复填充文本，属于合成负载，不应描述为真实用户文本。相同 seed 和负载参数在不同并发和实验 label 下可复现同一组请求。

## 真实负载：ShareGPT

[vLLM 官方压测文档](https://docs.vllm.ai/en/latest/benchmarking/cli/)提供 ShareGPT 负载示例。这里使用同一公开数据文件，读取每个会话的第一条 human 消息，作为独立的单轮请求；不会重放完整多轮会话。

数据较大，请按需自行下载一次：

```bash
mkdir -p benchmarks/data
curl -fL --retry 3 \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
  -o benchmarks/data/sharegpt.json

python benchmarks/benchmark_matrix.py \
  --model qwen2.5-7b --tokenizer "$TOKENIZER" \
  --dataset sharegpt --dataset-path benchmarks/data/sharegpt.json \
  --requests 128 --concurrency 1 4 8 \
  --input-lens 128 512 1024 --length-tolerance 0.2 \
  --output-lens 128 --seed 0 --label tp2-sharegpt
```

真实数据按输入长度的 ±20% 分桶，不做截断，也不为了凑数量重复抽样。记录实际长度分布和数据文件哈希。某个桶样本不足时，降低 `--requests` 或扩大 `--length-tolerance`。ShareGPT 是经过清洗筛选的公开对话子集，不能代表所有生产流量；具体文件说明见[数据卡](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)。

## 其他公开数据与自定义 JSONL

| 数据 | 适合用途 | 本项目使用方式 |
|---|---|---|
| ShareGPT | 真实单轮聊天文本与自然长度分布 | 已原生支持上述 JSON 文件 |
| [LongBench v1](https://huggingface.co/datasets/zai-org/LongBench) | 中英文长文问答、摘要等长输入场景 | 先将任务模板生成的完整 prompt 转为 JSONL；只选满足当前上下文预算的样本。改变模板、筛选或截断后，不应声称得到官方 LongBench 分数 |
| [OpenOrca / MLPerf Llama2 参考任务](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md) | 参考标准化任务的数据处理与实验方法 | 本脚本未实现 MLPerf 规则、LoadGen 或准确率验收，使用同一数据也不构成合规 MLPerf 成绩 |

LongBench 官方还提供直接下载原始数据的方式；使用原始 JSONL 可以避免旧数据集加载脚本与新版 `datasets` 的兼容问题。将你的任务输入整理为每行一条：

```json
{"id":"sample-001","prompt":"请阅读以下材料并回答问题：……"}
```

随后将 ShareGPT 示例中的数据参数替换为 `--dataset jsonl --dataset-path /path/to/prompts.jsonl`。脚本只测这些 prompt 的服务性能，不读取参考答案计算质量分数。

## 公平比较 TP=2 与 PP=2

1. 使用同一模型目录、dtype、上下文限制、token budget、并发上限和显存比例。
2. 两组都关闭 prefix cache 和 CUDA Graph，先隔离 TP/PP 差异；缓存或图优化可另开独立实验。
3. 不同时启动两套服务。每次切换后重新预热。
4. 使用同一 seed、tokenizer、数据文件和负载参数，只修改 `--label`（如 `tp2` / `pp2`）。核对对应 case 的 workload hash 一致。
5. 正式实验建议重复至少三次，保留所有轮次、错误率和输出长度。若输出明显变短或有错误，不能仅凭吞吐上升判断性能提升。
6. 保存服务端完整启动命令、git commit、GPU/驱动/PyTorch/NCCL 版本、GPU 拓扑及环境变量；客户端元数据不能自动证明远程服务端配置。

这是最大并发受限的闭环批量压测，不是固定 QPS 或 Poisson 到达流量。输入准备和 token 统计放在计时区间外；仍需留意客户端 CPU、网络及服务端逐步调试日志对测量的影响。

## 输出与验证

每次执行在 `--output-dir` 下创建独立目录，写入 `summary.json`、`summary.csv` 和逐请求 `requests.jsonl`。保留这些文件再比较结果；`--dry-run` 的指标为空，不是实测结果。

本脚本测试不启动模型：

```bash
python -m unittest discover -s benchmarks -p 'test_benchmark_matrix.py' -v
```

旧的 `profile_tp_pp.py` 保留用于复现历史实验，但其无 tokenizer 时按最大输出预算计数的结果不适合作为正式 token 吞吐；新实验使用本脚本。
