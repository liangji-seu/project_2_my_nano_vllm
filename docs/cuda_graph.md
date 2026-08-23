# 纯 Decode FULL CUDA Graph

## 支持边界

当前实现只捕获纯 Decode 的完整 Qwen2 模型 forward：

```text
input_ids / positions / attention metadata 固定缓冲
                         ↓
embed → 28 × (Attention + MLP) → final norm
                         ↓
                    hidden_states
```

`lm_head → argmax → D2H` 留在图外，与 vLLM 把采样作为模型 forward 后处理的
边界一致。Prefill、Chunked Prefill、混合 Prefill/Decode 和 mock CPU 路径继续
eager 执行；当前没有 PIECEWISE CUDA Graph。

纯 Decode 的判定条件为：

- batch 中每个请求进入本轮前已经有历史 KV；
- 每个请求本轮恰好调度 1 token；
- `num_actual_tokens == num_reqs` 且 `max_query_len == 1`。

## 固定地址与图的 key

`GPUModelRunner` 启动时已经预分配 `input_ids`、`positions`、`seq_lens`、
`query_start_loc`、`slot_mapping` 和 `block_table` 的 CPU/GPU 工作缓冲。每轮只
覆写这些固定地址的内容，因此 CUDA Graph replay 可以继续读取新 batch 数据。

图缓存 key 为：

```text
(num_tokens, num_reqs, max_seq_len_bucket)
```

当前没有把请求数 padding 到 capture bucket，所以不同 `num_reqs` 使用不同图。
Attention Triton kernel 的 `max_seq_len` 是录图后的固定 kernel 参数；实现将它
按默认 256 tokens 向上取整。同一桶内真实长度仍由 `seq_lens` GPU tensor 提供，
超过真实长度的 K/V tile 由 Attention mask 屏蔽。

第一次遇到一个 key 时执行：

```text
旁路 CUDA stream warmup
        ↓
捕获完整 model forward
        ↓
立即 replay 一次，得到本轮有效输出
```

后续相同 key 直接 `CUDAGraph.replay()`。wrapper 会记录所有模型输入 tensor 的
地址；地址改变时拒绝 replay，避免静默读取旧数据。

## 参数

CUDA Graph 默认启用：

```bash
my_vllm serve --model /path/to/Qwen2.5-7B-Instruct
```

可用参数：

```text
--disable-cuda-graph
--cuda-graph-seq-len-bucket-size 256
--cuda-graph-num-warmups 1
```

需要对照 eager 输出或性能时使用 `--disable-cuda-graph`。桶越小，Attention
无效 mask 计算越少，但会捕获更多图并增加首次捕获成本；桶越大则相反。

## 已验证内容

- CUDA Graph 独立测试：首次 capture 后修改同地址输入，replay 能消费新值；
- 完整测试：服务器 CUDA 环境 `27 passed`；
- Qwen2.5-7B 单卡：相同 greedy 请求连续两次 CUDA Graph 输出一致；
- Qwen2.5-7B 单卡：CUDA Graph 与 `--disable-cuda-graph` 输出逐 token 一致；
- Qwen2.5-7B 单卡：4 个并发请求均正确完成，未发生 KV/请求串扰。

当前测试证明了执行闭环和输出一致性；CUDA Graph 相对 eager 的 TTFT、TPOT 与
吞吐收益应在后续 profile 分支复用固定 256-request workload 单独测量。
