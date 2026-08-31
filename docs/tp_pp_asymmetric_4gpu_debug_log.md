# 非对称四卡环境下 TP/PP 开发与问题排查日志

## 1. 背景与目标

本日志记录 my-vLLM 在一台非对称 NUMA、无 GPU P2P 的四卡服务器上验证 Qwen2.5-7B TP/PP 时遇到的问题、排查过程、根因和最终修复。

验证目标不是只让四个进程成功启动，而是确认以下完整链路成立：

1. Qwen2.5-7B 的模型层和权重按 TP/PP rank 正确切分。
2. KV Cache 根据每个 PP stage 的本地 layer 集合独立构造，并使用全局统一 block 数。
3. EngineCore 使用异步 `step_with_batch_queue` 维护多个在途 micro-batch。
4. ModelRunner V2 将 `execute_model()` 和 `sample_tokens()` 拆开。
5. PP 最后 stage 采样后，将 token 通过独立 stream/communicator 旁路广播给前面 stage。
6. 在 TP=2/PP=2 和 TP=1/PP=4 下均能保持各 rank 的请求状态一致。
7. 使用同一份固定 workload 比较两种并行策略。

本轮没有启用 CUDA Graph，也没有启用 prefix cache，避免其他优化掩盖 TP/PP 本身的行为。

## 2. 服务器环境

### 2.1 软件与模型

- 模型：Qwen2.5-7B，BF16
- 模型结构：28 layers、hidden size 3584、28 Q heads、4 KV heads
- GPU：4 × NVIDIA GeForce RTX 4080 SUPER 32 GiB
- PyTorch：2.7.1+cu126
- CUDA：12.6
- NCCL：2.26.2
- 分支：`TP_PP/profile/qwen2.5_7B`
- 模型目录：`/root/autodl-tmp/liangji`
- 项目目录：`/root/autodl-tmp/work/project_2_my_nano_vllm`

### 2.2 非对称硬件拓扑

这台机器不是“两张卡一个 NUMA、另外两张卡另一个 NUMA”的对称布局：

- GPU0 位于 NUMA0。
- GPU1、GPU2、GPU3 位于 NUMA1。
- GPU1↔GPU2 为 PIX。
- GPU2↔GPU3 为 PXB。
- GPU0 与 GPU1/2/3 之间均为 SYS，需要跨 NUMA。
- `nvidia-smi topo -p2p` 显示四卡之间 P2P read/write 均不受支持。
- 机器没有 NVLink/NVSwitch。

因此，不可能把四张卡切成两个都位于各自 NUMA 内的 TP=2 小组。至少有一个 TP 组必须跨 NUMA，并且 TP all-reduce 不能走 GPU P2P。

这会直接影响并行策略：

- TP 会在每层的 RowParallelLinear 等位置频繁执行集合通信。
- PP 主要在 stage 边界传递 activation。
- 在无 P2P 环境里，深 PP 可能比 TP+PP 更有优势，即使深 PP 会增加流水线 bubble。

## 3. 测试前的代码状态确认

服务器迁移后，Git HEAD 仍停留在较早的 `46f6239`，但工作区已经包含通过文件同步传入的异步实现。先确认这些改动确实是预期代码，而不是服务器上的未知修改：

- `CUDAGraph` 在本轮配置中关闭。
- `EngineCore._step_with_batch_queue()` 已存在。
- `GPUModelRunner.execute_model()` 只保存 forward state。
- `GPUModelRunner.sample_tokens()` 独立完成采样。
- `PPTokenHandler` 使用单独 CUDA stream 和 sibling NCCL communicator。
- `MultiprocExecutor.collective_rpc(non_block=True)` 返回 Future。
- response MQ 由单线程按 RPC 广播顺序回收，避免两个 Future 交叉取错响应。

基础回归测试结果为 24 passed。随后把异步实现提交为：

- `9d47532 Add async PP batch queue and ModelRunner V2 sampling`

## 4. 异步 PP 数据流

### 4.1 EngineCore batch queue

当 `pipeline_parallel_size > 1` 时：

```text
batch_queue_size = pipeline_parallel_size
```

每次提交一个 batch 时，EngineCore 连续发出两条非阻塞 RPC：

```text
execute_model(scheduler_output, non_block=True)
sample_tokens(non_block=True)
```

然后把以下三项放进队列：

```text
(sample_future, scheduler_output, execute_future)
```

队列未满并且 Scheduler 还有独立工作时，EngineCore 继续提交 micro-batch；队列填满或没有更多可提交工作时，才核销最老 batch 的结果。

因此：

- PP=2 时最多维护 2 个在途 batch。
- PP=4 时最多维护 4 个在途 batch。

### 4.2 ModelRunner V2 拆分

`execute_model()` 负责：

1. 消费足够早以前收到的 PP token。
2. 用当前 SchedulerOutput 更新 CachedRequestState/InputBatch。
3. 构造扁平化模型输入和 attention metadata。
4. 执行本 stage forward。
5. 保存 `ExecuteModelState`，不在这里计算 logits 或返回采样结果。

`sample_tokens()` 负责：

1. 消费最近一次 `ExecuteModelState`。
2. 最后 PP stage 执行 lm_head、greedy argmax。
3. 最后 stage 把 token 广播给相同 TP lane 上的前面 PP stage。
4. 非最后 stage 在旁路 stream 上提交 broadcast receive。
5. 返回 ModelRunnerOutput；EngineCore 只消费最后 PP stage driver rank 的结果。

### 4.3 为什么前面 stage 仍要收到采样 token

虽然只有最后 stage 能计算 logits，但下一轮 decode 时所有 stage 都必须知道刚生成的 token：

- 第一个 stage 需要将 token 送入 embedding。
- 每个 stage 的 InputBatch/CachedRequestState 必须拥有一致的 token 序列。
- slot mapping、positions、seq_lens 和是否需要采样，都依赖一致的请求进度。

所以最后 stage 的 ModelRunnerOutput 返回 EngineCore，与 sampled token 的 PP 旁路广播，是两条同时存在的数据通路：

```text
最后 stage sampled token
  ├─ D2H / ModelRunnerOutput → EngineCore → Scheduler
  └─ GPU broadcast → 前面 PP stages 的本地请求状态
```

## 5. 非对称拓扑下的 TP=2/PP=2 映射选择

由于无法构造两个同 NUMA TP 组，测试了两种物理卡顺序。

### 5.1 映射 A

```bash
CUDA_VISIBLE_DEVICES=0,3,1,2
```

| PP stage | 逻辑 rank | 物理 GPU | TP 链路 |
|---|---|---|---|
| stage 0，layers 0–13 | 0,1 | GPU0,GPU3 | SYS，跨 NUMA |
| stage 1，layers 14–27 | 2,3 | GPU1,GPU2 | PIX，同 NUMA |

结果：496.78 output tok/s。

GPU0/3 平均利用率约 96–98%，GPU1/2 约 71–72%。跨 NUMA 的 stage 0 成为瓶颈，stage 1 存在明显等待。

### 5.2 映射 B

```bash
CUDA_VISIBLE_DEVICES=1,2,0,3
```

把 PIX TP 组放在 stage 0，把跨 NUMA TP 组放在 stage 1。结果为 492.01 output tok/s，比映射 A 低约 0.97%。

差距不大，但在当前固定 workload 下映射 A 更好，因此正式 TP=2/PP=2 结果采用映射 A。

## 6. PP=4 首次正式运行失败

### 6.1 冒烟测试为什么通过

先使用以下小配置强制填充四级 batch queue：

```text
TP=1
PP=4
token budget=4
8 concurrent requests
8 output tokens/request
```

该测试 8/8 输出一致，没有出现死锁。

但是，这批请求交错调度。同一个请求的连续 decode 步之间插入了其他请求的 batch，因此在一枚旁路 token 延迟四次 execute 后被消费时，该请求的本地 output 状态不一定已经继续前进三步。

正式压测前的 warmup 是单请求连续生成 128 tokens。相同请求会连续进入每一个 decode step，恰好覆盖了更严格的状态排列，最终暴露问题。

这说明并发冒烟测试只能验证通信和基本请求映射，不能代替“单个长生命周期请求连续 decode”的回归用例。

### 6.2 错误现象

正式 PP=4 的第一个 warmup 请求运行到足够多的 decode step 后，rank 0/1/2 同时报错：

```text
RuntimeError: 请求 <req_id> 的 PP 延迟 token 状态分叉
```

随后同一批 RPC 的 `sample_tokens()` 又报：

```text
RuntimeError: sample_tokens 前必须成功执行 execute_model
```

第二个异常是级联结果，不是根因：`execute_model()` 已经在消费延迟 token 时失败，所以没有生成新的 ExecuteModelState，后续 `sample_tokens()` 自然无法继续。

这轮压测被立即停止，所有结果作废，没有写入正式性能数据。

## 7. 根因分析

### 7.1 PP token FIFO

非最后 stage 的 PPTokenHandler 初始化一个长度为 `pp_size` 的 FIFO：

```text
PP=4: [None, None, None, None]
```

每次 execute 开始：

1. 从队首弹出一项。
2. 在队尾追加空槽。
3. 如果弹出的是 PendingRecv，则让主 stream 等待对应 CUDA event。

每次 sample：

1. 在旁路 stream 上提交 broadcast receive。
2. 保存 sampled tensor、request ids、should_sample。
3. 同时保存接收时每个请求的 `output_lengths`。

`output_lengths` 的语义是：这枚新 token 应该被写入请求 output token 数组的哪个位置。

例如接收 token `x` 时：

```text
expected_len = L
x 的稳定目标位置 = output_token_ids[L]
```

### 7.2 PP=2 下旧逻辑看似成立

旧校验逻辑等价于：

```python
if len(output_token_ids) == expected_len:
    append(token)
elif (
    len(output_token_ids) == expected_len + 1
    and output_token_ids[-1] == token
):
    # SchedulerOutput 已经同步了同一个 token
    pass
else:
    raise RuntimeError("PP 延迟 token 状态分叉")
```

PP=2 只延迟两个 execute step。常见状态下，这枚 token 被消费时：

- 本地尚未同步它：长度仍为 L，直接 append。
- SchedulerOutput 已经同步它：长度为 L+1，而且它仍是最后一项。

因此旧逻辑在 PP=2 测试中没有报错。

### 7.3 PP=4 下 token 不再位于末尾

假设在 step 0 收到 token `x`：

```text
接收时：output = [...]
expected_len = L
```

由于 PP=4 延迟四次 execute 才消费它，期间 SchedulerOutput 可能依次同步：

```text
step 1: output = [..., x]
step 2: output = [..., x, y]
step 3: output = [..., x, y, z]
step 4: 开始消费旧的旁路 token x
```

此时：

```text
len(output) = L + 3
output[-1] = z
output[L] = x
```

状态其实完全正确，但旧逻辑要求：

```text
len == L + 1 且最后一个 token == x
```

所以它把“Scheduler 已同步 x 以及后续 token”的正常状态误判为分叉。

问题不在 NCCL broadcast，也不在 request id 串线，而在 CPU 持久状态的校验条件把 PP=2 的时间关系写死了。

## 8. 修复方案

### 8.1 使用稳定索引，不使用当前末尾

修复后的判断为：

```python
if len(output_token_ids) == expected_len:
    append(token)
elif (
    len(output_token_ids) > expected_len
    and output_token_ids[expected_len] == token
):
    # token 已经由 SchedulerOutput 同步；后面可以还有更多 token
    pass
else:
    raise RuntimeError("PP 延迟 token 状态分叉")
```

这个判断表达的是实际不变量：

```text
pending token 对应的位置永远是接收时的 output 长度 expected_len
```

它不关心消费时列表末尾已经推进到哪里，因此同时适用于 PP=2、PP=4 和更深 PP。

### 8.2 为什么不能只放宽成“长度更长就跳过”

不能写成：

```python
if len(output_token_ids) > expected_len:
    pass
```

因为这样会掩盖真正的 token 分叉。必须继续校验：

```text
output_token_ids[expected_len] == pending token
```

如果这个位置不同，说明 Scheduler 路径和 GPU 旁路路径确实对同一个 request 产生了不同 token，必须立即失败，不能静默继续推理。

### 8.3 新增回归测试

新增测试构造以下状态：

```text
pending token = 10
expected_len = 0
本地 output = [10, 11, 12]
```

测试要求：

- `_consume_delayed_pp_tokens()` 不重复追加 10。
- 不因为 10 不是最后一项而报错。
- 本地 output 仍严格保持 `[10, 11, 12]`。

修复后完整测试集由 24 项增加为 25 项：

```text
25 passed
```

修复提交：

- `868d63d Fix delayed token validation for deep pipelines`

## 9. PP=4 修复后正式结果

固定 workload：

- 256 requests
- concurrency 32
- 1024 prompt tokens/request
- 128 output tokens/request
- max model length 4096
- token budget 2048
- prefix cache 关闭
- CUDA Graph 关闭
- warmup 4，不计入正式计时

PP=4 每个 stage 持有连续 7 层：

| stage | GPU | 模型内容 | 局部权重 |
|---|---:|---|---:|
| stage 0 | 0 | embedding + layers 0–6 | 4.064 GiB |
| stage 1 | 1 | layers 7–13 | 3.049 GiB |
| stage 2 | 2 | layers 14–20 | 3.049 GiB |
| stage 3 | 3 | layers 21–27 + norm/lm_head | 4.064 GiB |

KV Cache：

- 每 rank 7 个 attention layers。
- 每 rank 的一个物理 block 为 229,376 bytes。
- 统一 block 数为 98,102。
- block 数由显存余量最小的首尾 stage 决定。

性能对比：

| 指标 | TP=2 / PP=2 | TP=1 / PP=4 | 变化 |
|---|---:|---:|---:|
| 总耗时 | 65.9614 s | 60.0607 s | -8.95% |
| 请求吞吐 | 3.88 req/s | 4.26 req/s | +9.79% |
| 输出吞吐 | 496.78 tok/s | 545.58 tok/s | +9.82% |
| 总 token 吞吐 | 4,470.98 tok/s | 4,910.23 tok/s | +9.82% |
| 平均 E2E | 8,221.96 ms | 7,492.58 ms | -8.87% |
| P95 E2E | 9,533.01 ms | 8,457.09 ms | -11.29% |

PP=4 GPU 指标：

| GPU / stage | 平均利用率 | 峰值显存 |
|---|---:|---:|
| GPU0 / stage 0 | 99.20% | 26,566 MiB |
| GPU1 / stage 1 | 99.75% | 25,616 MiB |
| GPU2 / stage 2 | 99.77% | 25,616 MiB |
| GPU3 / stage 3 | 77.17% | 26,564 MiB |

## 10. 结论与经验

### 10.1 并行策略必须结合物理拓扑

“TP 通常比 PP 更高效”不能脱离硬件条件讨论。在支持 NVLink/NVSwitch 的机器上，TP all-reduce 可以非常快；在本机无 P2P、一个 TP 组必须跨 NUMA 的条件下，每层集合通信代价很高，PP=4 最终比 TP=2/PP=2 快 9.82%。

### 10.2 PP 深度是状态机正确性的一个维度

PP=2 正确不代表 PP=4 正确。任何用“前一拍、后一拍、最后一项”描述的逻辑，都要检查是否隐含写死了 pipeline depth。

更稳健的做法是保存并验证稳定标识：

- request id
- 接收时 output length
- token 在请求内的确定索引
- CUDA event 对应的具体 buffer

### 10.3 两条 token 通路必须用位置不变量对账

SchedulerOutput 同步和 GPU 旁路广播可能以不同时间到达同一个 Worker，但最终必须落在同一个 request 的同一个 token index。比较当前列表长度或最后一项不够稳健；比较确定索引才能区分“正常的到达顺序不同”和“真正的数据分叉”。

### 10.4 测试必须同时覆盖并发和单请求长生命周期

本轮并发小压测通过、单请求 warmup 失败，说明两者覆盖的状态排列不同。后续 PP 测试至少应包含：

1. 多请求、低 token budget，强制填满 batch queue。
2. 单请求连续 decode，覆盖同一请求跨越完整 PP 延迟窗口。
3. 不同 request 长度混合，覆盖 finished request 与 pending receive 相遇。
4. PP=1、PP=2、PP=4 参数化测试。

### 10.5 失败实验必须与正式结果隔离

第一次 PP=4 正式运行在 warmup 阶段失败，没有产生可用性能数据。修复、增加测试并重新启动全新服务后，才重新执行完整 workload。报告中的 PP=4 指标只来自修复后的成功运行。

## 11. 最终状态

- 异步调度实现提交：`9d47532`
- TP=2/PP=2 实验记录：`981f2d5`
- PP>2 延迟 token 修复与 PP=4 结果：`868d63d`
- 回归测试：25 passed
- 正式请求：256/256 完成
- 服务器服务进程已停止
- 四张 GPU 均回落到 1 MiB、0% utilization
