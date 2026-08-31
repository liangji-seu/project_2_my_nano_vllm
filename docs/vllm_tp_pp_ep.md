# vLLM TP、PP、EP 与异步调度源码理解

> 本文总结对 vLLM V1 Engine 中同步/异步 Scheduler、`step_with_batch_queue`、
> ModelRunner V1/V2，以及 TP、PP、EP 通信与调度关系的理解。

## 一、总体心智模型

三种并行解决的是不同维度的问题：

```text
EngineCore / Scheduler 时间维度
└── PP：模型按 layer 切成多个 stage，让多个 batch 在 stage 间流水
    └── 每个 PP stage 的一次 model.forward
        ├── TP：一个 layer 内的权重和计算横向分片
        └── EP：MoE token 按 expert 动态 dispatch、计算、combine
```

可以概括为：

- TP 主要解决“一个 layer 如何由多张卡共同计算”。
- PP 主要解决“不同 layer stage 如何让多个 batch 流水执行”。
- EP 主要解决“MoE experts 如何分布到不同 rank，以及 token 如何发往目标 expert”。

TP 的主要复杂度集中在模型层算子中的权重分片和 collective。PP 的模型切分位于
模型定义中，而主要运行时复杂度集中在 Scheduler、EngineCore batch queue、Worker
P2P 和 ModelRunner sampled-token 状态同步之间。EP 的主体位于 MoE layer 内部，
但 EP 跨 DP rank 后还需要额外的执行对齐。

## 二、TP：单次 forward 内部的算子并行

Scheduler 只产生一个逻辑 batch，Executor 把相同 `SchedulerOutput` 广播给所有
TP workers。每个 worker 处理相同 batch，但只持有相应的权重 shard。

典型并行算子包括：

```text
ColumnParallelLinear
RowParallelLinear
QKVParallelLinear
MergedColumnParallelLinear
VocabParallelEmbedding
ParallelLMHead
```

典型计算过程：

```text
QKV projection
  完整权重按输出维切分
  → 每个 TP rank 计算部分 Q/K/V 或部分 attention heads

O projection
  输入按 TP rank 分片
  → 每个 rank 计算部分输出
  → all-reduce / reduce-scatter 汇总

MLP
  gate/up projection 按中间维切分
  → 每个 rank 计算部分激活
  → down projection 后 collective 汇总
```

所以 ModelRunner 表面上只调用：

```python
model_output = self.model(...)
```

真正的 TP 分片和 `all-reduce`、`all-gather`、`reduce-scatter` 等通信发生在
DecoderLayer、Attention、Linear、MLP 等模型算子内部。EngineCore 和 Scheduler
通常不感知 TP 的细节。

## 三、PP：模型空间切分与 batch 时间流水

### 3.1 模型层切分

假设模型有 `L` 层，PP=3：

```text
PP0: embedding + layers [0, n)
PP1: layers [n, m)
PP2: layers [m, L) + norm + lm_head
```

模型定义通过当前 PP rank 得到 `start_layer/end_layer`：

- first PP rank 负责 embedding；
- 中间 rank 消费上游 `IntermediateTensors`；
- 每个 rank 只执行自己的 layer 范围；
- 非尾 rank 返回 `hidden_states/residual`；
- last PP rank 才执行 final norm、lm head、logits 和采样。

这只是 PP 的空间切分。要真正形成流水，还需要 Worker 的 activation P2P 和
EngineCore 的多 batch 在途调度。

### 3.2 Activation P2P：`isend_tensor_dict/irecv_tensor_dict`

每个 batch 的中间激活逐 stage 向后传递：

```text
PP0 ── hidden_states ──> PP1 ── hidden_states ──> PP2
```

这是相邻 stage 之间的 point-to-point 通信。

发送端的主要过程：

```text
ModelRunner forward
→ 返回 IntermediateTensors
→ GPUWorker 调用 isend_tensor_dict
→ 先发送 tensor dict metadata
→ 对 GPU tensor 发起 torch.distributed.isend
→ 保存通信 handles
→ 当前 execute_model 返回
```

下一次 `execute_model` 开头会等待上一次 send handles，防止 activation buffer
还未发送完成就被下一批复用或覆盖。非阻塞 send 并不意味着永远不等待，而是把
等待推迟到下一次执行的安全边界，从而创造通信与其他工作重叠的机会。

接收端的主要过程：

```text
收到 tensor metadata
→ 按 shape/dtype/device 分配接收 tensor
→ 发起 torch.distributed.irecv
→ 将 tensor、handles、postprocess 封装为 AsyncIntermediateTensors
→ 真正访问 .tensors 时才 wait handles
→ 必要时执行 TP all-gather 后处理
→ 开始本 stage forward
```

因此 `irecv_tensor_dict` 并非从头到尾完全不阻塞：metadata 接收可能等待；真正
的大 tensor 数据使用异步 `irecv`，并通过 `AsyncIntermediateTensors` 把等待
推迟到 activation 真正被模型使用时。

逻辑依赖始终存在：

```text
PP1.forward(B) 必须等待 PP0 产生 activation(B)
```

这是 PP 正常的数据依赖。稳定流水时，上游计算 B 的时间与下游计算 A 的时间重叠：

```text
时间 ─────────────────────────────────────────>

PP0: forward A    forward B    forward C
PP1:              forward A    forward B    forward C
PP2:                           forward A    forward B
```

TP+PP 下还可以只沿每条 PP lane 发送 activation 的一部分，再在接收 stage 内通过
TP all-gather 恢复完整 tensor，以避免多个 TP rank 重复发送相同数据。

## 四、EngineCore 为什么能发很多 batch，而 ModelRunner 仍一次只跑一个 batch

`step_with_batch_queue` 可以连续提交：

```text
execute_model(A)
sample_tokens(A)
execute_model(B)
sample_tokens(B)
...
```

`non_block=True` 只表示：

```text
RPC 已经放入 Worker 广播队列
Future 已经返回给 EngineCore
```

它不表示同一个 Worker 同时进入了多个 ModelRunner forward。每个 Worker 的 RPC
busy loop 仍然单线程、按 FIFO 执行：

```text
execute(A) → sample(A) → execute(B) → sample(B)
```

因此 ModelRunner 可以只维护一个 `execute_model_state`：

```text
execute_model(A)
→ 保存 logits/hidden states/input metadata

sample_tokens(A)
→ 消费并清空 execute_model_state

execute_model(B)
→ 才能写入下一份 state
```

这里有三个必须区分的状态：

```text
RPC 已发射
≠ Worker 已经开始执行
≠ 单个 ModelRunner 并发执行多个 forward
```

PP 的并行来自“不同 PP stage 同时处理不同 batch”，不是一个 ModelRunner 同时跑
两个 forward。

## 五、同一个 request 能否连续进入两个 decode batch

### 5.1 同步 Scheduler

同步 Scheduler 发射一个 decode token 后会乐观增加 `num_computed_tokens`。在采样
结果返回并追加到 request 之前：

```text
num_tokens == num_computed_tokens
num_new_tokens == 0
```

因此同一个 request 通常不能连续进入两个 decode batch；必须先处理前一批采样
结果。Chunked prefill 不同，因为 prompt 中仍可能有未计算 token，所以同一个
request 可以连续进入多个 prefill batch。

### 5.2 AsyncScheduler + ModelRunner V1

AsyncScheduler 在发射 decode 后增加 output placeholder，表示“真实 token 尚未返回，
但假定这个输出位置将存在”。因此 Scheduler 可以在前一采样结果尚未回到
EngineCore 时再次调度相同 request。

ModelRunner V1 不设置 PP decode cadence，所以同一个 request 可以连续进入多个
逻辑 batch。

### 5.3 AsyncScheduler + ModelRunner V2

V2 发射 decode 后设置：

```text
next_decode_eligible_step = current_step + pp_size
```

所以同一个 request 在 `pp_size` 步内不会再次 decode：

```text
step T:     req0 decode
step T+1:   req0 被 throttle
...
step T+P:   req0 最早重新 eligible
```

这里表示“最早在 T+P 重新出现”，不表示 request 必须刚好在 T+P 被调度。

## 六、同步 PP 与 sampled token 回流

last PP stage 产生 sampled token，但 first/middle stages 在后续 decode 时也需要真实
token。

同步 Scheduler 的路径是：

```text
last PP stage sample
→ Worker response
→ Executor Future
→ EngineCore
→ scheduler.update_from_output
→ 下一次 SchedulerOutput 携带新 token
→ 所有 PP workers 更新本地 request state
```

因为同步 Scheduler 本来就会等采样结果，所以这条 CPU round trip 不会额外破坏
它的语义。

在 `V1 + PP + SyncScheduler` 下，非尾 rank 完成 forward 后返回
`IntermediateTensors`，其 `sample_tokens` 没有本地 logits/state，也不需要接收
GPU sampled-token broadcast，因此基本直接返回。它可以继续处理下一个 batch：

```text
PP0: forward A    forward B    forward C
PP1:              forward A    forward B
PP2:                           forward A/sample A
```

## 七、V1 + PP + AsyncScheduler 的主流 broadcast

AsyncScheduler 会在 output A 尚未回到 EngineCore 时发射后续 batch。如果同一个
request 连续进入 A、B，那么 B 的前级 PP stage 需要 A 采样出的真实 token，不能
只依赖 CPU 侧 placeholder。

V1 因此增加 GPU sampled-token broadcast：

```text
PP0 <──────────────┐
PP1 <── token A ───┤
PP2(last) ─────────┘
```

尾 stage 在 `sample_tokens(A)` 后从 PP group 广播 token；非尾 stage 在自己的
`sample_tokens(A)` 中进入匹配的 broadcast receive，并把 token 写入本地
`InputBatch`。

V1 的问题是：

- sampled-token broadcast 使用 PP 原本的 `device_group`；
- activation P2P 也使用这个 PP communicator；
- broadcast 被提交在默认/main CUDA stream 上；
- 后续 forward kernel 在同一 main stream 中排在 broadcast 后面。

因此即使 Python 的 `broadcast()` 可能在 NCCL 工作提交后返回，GPU 主流仍然存在：

```text
forward A
→ broadcast receive token A
→ forward B
```

时间线近似为：

```text
PP0: forward A ───────── wait/receive token A ─→ forward B
PP1:           forward A ─ wait/receive token A → forward B
PP2:                       forward A/sample A → broadcast A
```

所以 V1 的 PP 本身可以工作，但 `V1 + PP + AsyncScheduler` 中 sampled-token 回流
没有和非尾 stage 的下一批 forward 解耦，容易制造很大的 pipeline bubble。这是
“V1 未完整支持 PP async scheduling”的核心原因之一。

## 八、V2 旁路 sampled-token 通信

V2 没有取消 sampled-token broadcast，而是通过三个配套机制把等待尽量移出后续
forward 的关键路径：

1. 独立 CUDA `broadcast_stream`；
2. 相同成员、独立 NCCL communicator 的 sibling process group；
3. `pp_size` decode throttle + 固定延迟 FIFO 提交。

### 8.1 独立 stream 与 communicator

V2 中：

```text
main_stream:
    activation P2P、模型 forward、采样、状态更新

broadcast_stream:
    sampled token、num_sampled、num_rejected 的 broadcast

PP device_group:
    activation isend/irecv

PP broadcast_group:
    sampled-token broadcast
```

独立 stream 使 sampled-token 通信可以与后续 forward 重叠；独立 communicator
避免 sampled-token collective 与 activation P2P 因共用 NCCL communicator 而
相互串行化或产生顺序耦合。

尾 stage 的依赖方向是：

```text
main stream 产生 sampled token
→ broadcast stream 等 main stream
→ broadcast stream 发送 token
```

main stream 不会在采样后立即反向等待 broadcast 完成。

非尾 stage 在 `sample_tokens(T)` 中：

```text
在 broadcast stream 上提交 receive
→ 接收 sampled_tokens
→ 接收 num_sampled/num_rejected
→ record event(T)
→ 将 buffer、event、request slot 映射保存为 PendingRecv(T)
→ 放入 FIFO
→ 返回，不让 main stream 立即等待
```

### 8.2 FIFO 为什么预填 `pp_size` 个 `None`

非尾 stage 的 FIFO 初始为：

```python
deque([None] * pp_size)
```

每次 `execute_model` 开头：

```text
pop 最老槽位
append 一个新的 None，预留本 step 的接收槽位
```

每次 `sample_tokens` 中如果本 batch 有以后需要的 sampled output：

```text
把刚预留的队尾 None 替换为 PendingRecv(current_step)
```

假设 PP=2：

```text
初始:               [None, None]

execute T:          pop None, append None
sample T receive:   [None, token(T)]

execute T+1:        pop None, append None
sample T+1 receive: [token(T), token(T+1)]

execute T+2:        pop token(T), append None
                     ↑
                     此时消费 step T 的采样结果
```

FIFO 是固定延迟 `pp_size` 步的 sampled-output 提交队列，而不是“未来碰到相同
request 时再按 req_id 查找 event”的缓存。

### 8.3 非尾 rank 的 `wait_event` 发生在什么时候

非尾 rank 在 `sample_tokens(T)` 中只是旁路接收并记录 event。真正的：

```python
main_stream.wait_event(event_T)
```

发生在未来 `execute_model(T + pp_size)` 开头：

```text
execute_model(T+P)
├── update_pp_decode_requests
│   ├── 从 FIFO 弹出 PendingRecv(T)
│   ├── 过滤已 abort/free/reuse 的 request slots
│   ├── main_stream.wait_event(event_T)
│   └── postprocess_sampled(token_T)
├── finish/free/add/update 当前 SchedulerOutput
├── prepare inputs
└── forward(T+P)
```

`wait_event()` 通常不会让 CPU 在 Python 这一行忙等，而是在 CUDA main stream 中
插入跨 stream 依赖：

```text
broadcast stream 完成 receive(T)
→ event(T)
→ main stream 读取 token(T)
→ 更新本地 request state
→ 后续 forward
```

如果 event 已完成，main stream 几乎直接通过；如果尚未完成，GPU main stream
执行到该依赖节点时才停下来等待。

因此 V2 不是消灭等待，而是把等待从“每个 batch 采样后立即等待”推迟
`pp_size` 步：

```text
V1 暴露等待
≈ sampled-token 通信时间

V2 暴露等待
≈ max(0, sampled-token 通信时间 - 中间 P 步可重叠的计算时间)
```

### 8.4 FIFO 与 Scheduler throttle 如何配合

两者分别保证：

```text
FIFO:
    step T 的采样结果固定在 T+P 提交到非尾 rank 持久状态

Scheduler throttle:
    step T 中的同一个 request 在 T+P 之前不会再次 decode
```

它们共同保证：

```text
sampled-token 状态提交时间
<=
相同 request 下一次可能使用时间
```

如果 request 恰好在 T+P 再出现，`execute_model(T+P)` 会先提交旧 token，再处理
当前 SchedulerOutput 和准备输入。如果 request 更晚才出现，旧 token 已经提前写入
持久状态，可以直接使用。

### 8.5 FIFO 元素与未来 batch 不需要一一对应

一个 `PendingRecv(T)` 代表 step T 整个 batch 的输出，例如：

```text
batch A:
row 0 → req0
row 1 → req1
row 2 → req2
```

它保存：

```text
sampled_tokens
num_sampled / num_rejected
idx_mapping
need_sampled_mask
slot generation snapshot
event
```

到 T+P 时，不管当前 batch 是否包含 req0/req1/req2，ModelRunner 都按 A 自己保存的
`idx_mapping` 把仍有效的输出写回各 request 的持久状态：

```text
PendingRecv(A)
    ├── 更新 req0 persistent state
    ├── 更新 req1 persistent state
    └── 更新 req2 persistent state
```

如果 T+P 当前只调度 req0，req1 的状态仍然会更新，等 req1 更晚出现时使用。如果
当前一个都没有，也会做固定延迟提交；这是一个保守但简单的 cadence 设计。

若 request 在等待期间 abort、free 或 slot 被复用，generation counter 会变化。
消费时会将陈旧 slot 过滤或把 `idx_mapping` 置为 `-1`，避免旧 token 写进新 request。
如果整个 PendingRecv 都已无效，可以直接丢弃而不等待 event。

### 8.6 V2 完整时间线

假设 PP=2：

```text
逻辑 step          T                   T+1                    T+2
请求              req0                req1                   req0 或其他

PP0 main:      forward(req0)      forward(req1)      wait_event(token T)
PP0 side:           recv token T ───────────────→ event T        │
                                                               postprocess
                                                               forward(T+2)

PP1 main:            forward(req0)
                     sample token T
PP1 side:               broadcast token T ─────→ event T
```

V2 的本质是：

```text
提前在旁路流启动通信
→ 中间 P 步主流继续 forward 其他 request
→ 最早需要旧 token 时插入 wait_event
→ 理想情况下 event 已经完成，等待被隐藏
```

## 九、EP 对 ModelRunner 调度的影响

### 9.1 基础 EP 主体仍在 MoE layer 内部

开启 EP 后，每个 rank 只持有部分 experts：

```text
hidden states
→ router / top-k
→ 按目标 expert dispatch tokens
→ EP all-to-all
→ 每个 rank 计算 local experts
→ combine / reduce-scatter / all-to-all
→ 恢复 token 原顺序
```

Scheduler 不负责决定 token 去哪个 expert。expert 路由由 GPU 上的 MoE layer 根据
当前 hidden states 动态计算，因此基础 EP 不改变：

- `schedule → execute → sample → update` 主循环；
- `step_with_batch_queue`；
- PP 在途 batch 数；
- V1/V2 sampled-token 回流；
- 同一个 request 的 decode cadence。

### 9.2 EP 跨 DP rank 后会改变执行协调

当 EP group 跨越多个 DP rank，不同 DP rank 可以有不同的本地 batch：

```text
DP0: 128 tokens
DP1: 64 tokens
DP2: 0 tokens
DP3: 32 tokens
```

但它们必须按相同顺序进入 MoE collective，否则某个 rank 不参与就可能导致其他
rank hang。ModelRunner 因此需要：

- 跨 DP 同步每个 rank 的 token 数；
- 协调 CUDA Graph mode 和 padding；
- 将 `num_tokens_across_dp` 传给 MoE kernel；
- 没有本地请求的 DP rank 仍运行 dummy forward；
- 全局协调 DP engine 是否还有未完成请求。

这改变的是 ModelRunner/EngineCore 的执行协调，不是本地 Scheduler 选择 request
的算法。

### 9.3 EP、TP、PP 的组合

EP group 通常位于每个 PP stage 内，不跨 PP stage。例如 TP=2、PP=2、DP=2：

```text
PP stage 0:
  DP0/TP0 ┐
  DP0/TP1 ├── EP group 0
  DP1/TP0 │
  DP1/TP1 ┘

PP stage 1:
  DP0/TP0 ┐
  DP0/TP1 ├── EP group 1
  DP1/TP0 │
  DP1/TP1 ┘
```

一个 batch 的执行层次是：

```text
PP0
├── dense/attention: TP collective
├── MoE: EP dispatch / expert compute / combine
└── activation P2P → PP1
                       ├── dense/attention: TP collective
                       ├── MoE: EP collective
                       └── logits/sample
```

EP 不改变 PP 的逻辑，但 expert 路由不均和 all-to-all 尾延迟可能让某个 PP stage
变慢，造成 stage imbalance 和额外 pipeline bubble。

EPLB 则在基础 EP 上增加 expert 热度统计、冗余 expert、logical-to-physical expert
映射和运行时 placement 更新。它仍不选择 request，但会改变 expert 权重放置和
执行时的路由映射。

## 十、最终总结

vLLM 的 PP 主要由四部分组成：

```text
1. 模型 layer 按 PP rank 切分
2. Worker 用 activation P2P 连接相邻 stage
3. EngineCore 用 batch queue 保持多个逻辑 batch 在途
4. sampled token 从尾 stage 回流到非尾 stage
```

其中 sampled-token 回流决定了异步 PP 是否真正高效：

```text
Sync Scheduler:
    token 经 EngineCore 回传，下次调度再下发

V1 Async:
    token 在 main stream 上立即 broadcast/receive
    → 后续 forward 被挡住，PP overlap 较差

V2 Async:
    side stream + sibling communicator 提前接收
    + FIFO 固定延迟 P 步提交
    + Scheduler 同 request throttle P 步
    → 将 token 通信和中间 P 步 forward 重叠
```

最简洁的结论是：

> vLLM 的 PP = layer 分段 + activation P2P + EngineCore 多 batch 在途流水 +
> autoregressive sampled-token 回流。V2 再用 decode throttle、旁路 broadcast 和
> 延迟 FIFO 提交，把 token 回流的等待从每个 batch 后面推迟到最早需要旧 token
> 的安全边界，并尽量用中间 `pp_size` 步的 forward 隐藏通信。

