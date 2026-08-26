# TP/PP 性能测量

在至少两张 GPU 的服务器上，从 `TP_PP/profile/qwen2.5_7B` 分支执行。当前
Qwen2.5 模型的 TP/PP 权重切分仍在开发中，因此先使用该入口统一记录配置与
HTTP 服务端指标；不要将单卡结果标为 TP/PP 结果。

```bash
python -m my_vllm.cli serve \
  --model /path/to/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --tp-size 2 \
  --pp-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048
```

服务完成预热后，在另一终端运行：

```bash
python benchmarks/profile_tp_pp.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --requests 256 \
  --concurrency 16 \
  --max-tokens 128
```

脚本输出端到端延迟和输出字符吞吐。当前 HTTP 接口尚未提供流式首 token 回调，
因此 `ttft_ms` 明确为 `null`；在流式接口接入前，不应将其作为 TTFT 数据。
