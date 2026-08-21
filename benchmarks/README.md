# Qwen2.5-7B Baseline Benchmark

固定工作负载：256个请求，每条请求的在线服务实际输入为1024 tokens，服务端
`max_model_len=4096`。默认每条生成128 tokens并忽略EOS，避免模型提前停止导致
不同优化版本的工作量不一致。

生成固定请求文件（只需执行一次，生成后应提交到Git）：

```bash
python benchmarks/generate_qwen2_5_7b_workload.py \
  --model /home/liangji/huggingface/Qwen2.5-7B-Instruct-1M \
  --output benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl
```

启动Baseline服务：

```bash
CUDA_VISIBLE_DEVICES=0 my_vllm serve \
  --model /home/liangji/huggingface/Qwen2.5-7B-Instruct-1M \
  --host 127.0.0.1 --port 13311 \
  --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --num-gpu-blocks 4097 \
  --disable-prefix-caching
```

执行压测：

```bash
python benchmarks/benchmark_online_serving.py \
  --workload benchmarks/workloads/qwen2_5_7b_256req_1024prompt_128output.jsonl \
  --concurrency 16 \
  --output benchmarks/results/qwen2_5_7b_baseline_c16.json
```

`TTFT` 从FastAPI收到请求开始，到EngineCore取得第一个采样token结束；`TPOT`
为首token之后的生成耗时除以剩余输出token数。HTTP接口仍是非流式，因此不能
用客户端收到响应的时间代替TTFT。
当前Baseline实测报告见：
`benchmarks/results/qwen2_5_7b_baseline_256req_c16.md`。

用于后续自动汇总和版本对比的完整实验配置见：
`benchmarks/results/qwen2_5_7b_baseline_256req_c16_manifest.json`。
