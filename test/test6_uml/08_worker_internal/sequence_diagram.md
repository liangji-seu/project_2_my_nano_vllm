# 08 · GPUModelRunner 时序图

> **一次 `execute_model()` 的完整内部流水线**，以及"采样延迟"（grammar 约束）的变体。

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant MR as GPUModelRunner
    participant BS as CachedRequestState<br/>(输入 batch)
    participant M as torch model
    participant EC as EngineCore

    W->>MR: execute_model(scheduler_output)

    Note over MR: ===== 1. 预处理 =====
    MR->>BS: _update_states(scheduler_output)
    BS-->>MR: 增量更新 block_ids / computed / positions
    MR->>MR: _prepare_inputs(...)
    Note over MR: 拼好 input_ids、positions、<br/>slot_mappings、attention_metadata
    MR->>MR: _determine_batch_execution_and_padding()
    Note over MR: 选 cudagraph 模式 / padding

    Note over MR: ===== 2. 前向 =====
    MR->>M: _model_forward(input_tensors)
    M-->>MR: logits (未采样分布)

    Note over MR: ===== 3. 采样 =====
    alt 无 grammar 约束 (直接采样)
        MR->>MR: _sample() 采样 token
        MR-->>W: ModelRunnerOutput
    else 有 grammar 约束 (结构化输出)
        MR->>MR: 状态存进 execute_model_state
        MR-->>W: return None
        Note over W,EC: EngineCore 侧拿 grammar bitmask 后再采样
        EC->>W: sample_tokens(grammar_output)
        W->>MR: sample_tokens(grammar_output)
        MR->>MR: 用 bitmask 约束后采样
        MR-->>W: ModelRunnerOutput
    end

    Note over MR: ===== 4. 收尾 =====
    MR->>MR: _bookkeeping_sync() 同步 CUDA 状态
```

## 关键点

1. **四段流水线**：预处理 → 前向 → 采样 → 收尾。前向只产出 **logits（未采样分布）**，
   采样是独立一步。
2. **采样延迟的核心动机**：结构化输出（JSON schema / grammar）需要在采样前把
   grammar bitmask 传进来，而 bitmask 由 Scheduler 侧算好（`get_grammar_bitmask`），
   所以 `execute_model` 先算完 logits 就返回 None，等 bitmask 到位再 `sample_tokens`。
3. **增量状态**：`_update_states` 只更新本轮新增的部分，旧 block 的状态直接复用，
   这是 V1 相比 V0 每步全量重建输入的性能关键。
4. **cudagraph 选择**：`_determine_batch_execution_and_padding` 根据 batch 形状决定
   是否走 CUDA Graph 路径，避免动态 shape 的 kernel 启动开销。
