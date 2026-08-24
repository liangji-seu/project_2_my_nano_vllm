# 纯 Decode FULL CUDA Graph

## 架构与启动顺序

当前实现按照 vLLM 的职责边界拆成以下组件：

```text
BatchDescriptor
      ↓
CUDAGraphDispatcher ── 合法 (mode, key) 描述符库
      ↓
CUDAGraphWrapper    ── capture / replay
      ↓
CUDAGraphEntry      ── graph / output / input addresses
```

完整启动顺序为：

```text
GPUModelRunner.__init__
    └── 构造空 CUDAGraphDispatcher

GPUModelRunner.load_model
    └── 用 CUDAGraphWrapper 包装 model_forward

GPUModelRunner.initialize_kv_cache
    ├── 分配并绑定分页 KV Cache
    └── initialize_cudagraph_keys()
          └── 建立合法 (FULL, BatchDescriptor) 库

GPUModelRunner.capture_model
    └── 遍历 Dispatcher 的全部合法描述符
          ├── _dummy_run(mode=NONE)  eager warmup
          └── _dummy_run(mode=FULL)  触发 Wrapper capture

GPUModelRunner.execute_model
    ├── Dispatcher 根据真实 batch 选择 mode/key
    └── Wrapper
          ├── FULL + entry 命中：replay
          └── NONE：eager
```

启动主动捕获结束后，Wrapper 会关闭 capture-on-miss。真实请求没有命中合法
描述符时直接 eager，不会在服务期间突然同步录制新图。

## 合法描述符库

`BatchDescriptor` 包含：

```text
(num_tokens, num_reqs, max_seq_len, is_uniform)
```

第一阶段只注册纯 Decode FULL key，因此：

```text
num_tokens == num_reqs
max_query_len == 1
is_uniform == True
```

默认 batch 档位为：

```text
1, 2, 4, 8, 16, 32
```

超过 `max_num_seqs` 的档位会在 Dispatcher 初始化时过滤。序列长度从
`cuda_graph_seq_len_bucket_size` 使用固定步长增长，并始终包含
`max_model_len`。例如 `bucket_size=256, max_model_len=4096` 时为：

```text
256, 512, 768, 1024, 1280, ..., 3840, 4096
```

真实 Decode 的最大序列长度向上匹配最近的合法档位。`seq_lens` 仍是每轮覆写
的固定地址 GPU tensor，档位超出真实长度的 K/V 范围由 Attention kernel mask。
当前没有实现请求数 padding，因此 3、5 等未配置 batch size 会回退 eager。

## 捕获边界与固定地址

捕获范围是完整 Qwen2 模型 forward：

```text
input_ids / positions / attention metadata 固定缓冲
                         ↓
embed → N × (Attention + MLP) → final norm
                         ↓
                    hidden_states
```

`lm_head → argmax → D2H` 留在图外。Prefill、Chunked Prefill、混合 batch 和
mock CPU 路径继续 eager；当前没有 PIECEWISE CUDA Graph。

`_dummy_run()` 与真实 `execute_model()` 复用 ModelRunner 预分配的
`input_ids`、`positions`、`query_start_loc`、`seq_lens`、`slot_mapping` 和
`block_table` 缓冲。Wrapper 在捕获时记录地址，replay 前再次核验，避免静默
读取已经换址的输入。

## 参数

CUDA Graph 默认启用：

```bash
my_vllm serve --model /path/to/Qwen2.5-7B-Instruct
```

相关参数：

```text
--disable-cuda-graph
--cuda-graph-capture-sizes 1 2 4 8 16 32
--cuda-graph-seq-len-bucket-size 256
--cuda-graph-num-warmups 1
```

档位越密，真实 batch 命中率越高，但启动捕获耗时和 CUDA Graph 元数据开销也
越大。需要对照 eager 输出或性能时使用 `--disable-cuda-graph`。
