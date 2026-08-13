# 04 · Scheduler 时序图

> **一次 `schedule()` 的三阶段**：RUNNING（续跑）→ WAITING（恢复被抢占 + 新请求 prefill）→
> 分配 KV block → 产出 `SchedulerOutput`。最后 `update_from_output()` 收尾。

```mermaid
sequenceDiagram
    autonumber
    participant EC as EngineCore
    participant S as Scheduler
    participant RQ as running/waiting<br/>队列
    participant KV as KVCacheManager

    EC->>S: schedule(throttle_prefills)

    Note over S: ==== Phase 0: 初始化预算 ====
    S->>S: current_step += 1
    S->>S: token_budget = max_num_scheduled_tokens
    S->>KV: new_step_starts() 清空上轮临时状态

    Note over S: ==== Phase 1: RUNNING 续跑 ====
    loop while 还有 token 预算 and 未遍历完 running
        S->>RQ: 取 running[req_index]
        alt 请求能继续 (decode / 续跑 prefill chunk)
            S->>S: num_new_tokens = with_spec - computed
            S->>KV: allocate_slots(req, num_new_tokens)
            KV-->>S: new_blocks / 复用已有 blocks
            S->>S: scheduled_running_reqs.append(req)
        else 预算不够 / 超出约束
            S->>S: _preempt_request(req) 踢回 waiting
            S->>RQ: waiting.prepend_request(req)
        end
    end

    Note over S: ==== Phase 2: WAITING (恢复 + 新请求) ====
    loop while (waiting 或 skipped_waiting 非空) and 还有预算
        S->>RQ: 从 waiting 弹出请求 req
        alt 状态 = WAITING_FOR_REMOTE_KVS (P/D 场景)
            S->>RQ: skipped_waiting.prepend_request(req) 暂跳过
        else 是之前被抢占、要恢复的
            S->>KV: allocate_slots(...) 重新分配
            S->>S: scheduled_resumed_reqs.append(req)
        else 是全新请求 (prefill)
            S->>KV: allocate_slots(req, num_new_tokens)
            S->>S: scheduled_new_reqs.append(req)
        end
    end

    Note over S: ==== 收尾 ====
    S->>S: 合并 scheduled_new_reqs += scheduled_resumed_reqs
    S-->>EC: SchedulerOutput(三份名单 + req_to_new_blocks)

    Note over EC,KV: ===== 执行器跑完后 =====
    EC->>S: update_from_output(scheduler_output, model_output)
    S->>S: 更新每个 req 的 computed/output tokens
    S->>S: 完成/中止的请求 → free 其 KV blocks
    S->>KV: free(request) 回收 block
    S-->>EC: EngineCoreOutputs
```

## 关键点

1. **三份名单**：`scheduled_running_reqs`（续跑）、`scheduled_resumed_reqs`（被抢占恢复）、
   `scheduled_new_reqs`（新 prefill），一起写进 `SchedulerOutput` 交给执行器。
2. **KV 分配在调度时就完成**：`schedule()` 里直接调 `kv_cache_manager.allocate_slots()`，
   所以 `SchedulerOutput` 里已经带上了 `req_to_new_blocks`（每个请求新分配的 block）。
3. **抢占（preempt）**：token 预算不够时，把 running 里的请求踢回 waiting，等下一轮恢复。
4. **`WAITING_FOR_REMOTE_KVS`**：P/D 分离时，等远端 KV 传输的请求本轮跳过，不浪费预算。
