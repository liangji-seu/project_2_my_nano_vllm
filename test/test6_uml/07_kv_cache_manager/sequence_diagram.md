# 07 · KVCacheManager 时序图

> 一次 `allocate_slots()` 的三阶段分配，以及请求结束后的 `free()` 回收。

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant KM as KVCacheManager
    participant CO as KVCacheCoordinator
    participant FP as FreeKVCacheBlockQueue

    Note over S,FP: ===== 前缀缓存命中检查 =====
    S->>KM: get_computed_blocks(request)
    KM->>KM: 按 block_hash 查 prefix cache
    alt 前缀命中
        KM-->>S: (可复用 blocks, num_hits, 0)
    else 未命中
        KM-->>S: (empty, 0, num_miss)
    end

    Note over S,FP: ===== allocate_slots 三阶段 =====
    S->>KM: allocate_slots(request, num_new_tokens)

    rect rgb(240,248,255)
    Note over KM: 阶段1: 清理 comp 里多余的旧 block
    KM->>CO: remove_skipped_blocks() 释放滑动窗口外的 block
    CO->>FP: push(block) 归还
    end

    rect rgb(240,255,240)
    Note over KM: 阶段2: 处理前缀 token (comp + new_comp + ext_comp)
    KM->>CO: get_num_blocks_to_allocate(...)
    CO-->>KM: 需要分配的 block 数
    alt 空闲 block 不足 (required > available)
        KM-->>S: None (本轮不调度此请求)
    end
    end

    rect rgb(255,250,240)
    Note over KM: 阶段3: 为要算的新 token (new + lookahead) 分配
    KM->>CO: 分配新 block
    CO->>FP: pop() 取空闲 block
    FP-->>CO: KVCacheBlock
    CO-->>KM: 新分配的 blocks
    KM->>KM: 更新请求 block 映射 + 引用计数
    end

    KM-->>S: KVCacheBlocks (新 block 列表)

    Note over S,FP: ===== 请求结束后回收 =====
    S->>KM: free(request)
    KM->>KM: 每个 block ref_cnt -= 1
    KM->>CO: 引用归零的 block 归还
    CO->>FP: push(block)
```

## 关键点

1. **前缀缓存先查后分**：`get_computed_blocks()` 先看有没有现成 block 能复用；
   `allocate_slots()` 只给真正需要新算的 token 分配 block。
2. **三阶段分配**（源码 `kv_cache_manager.py:486` 注释原文）：
   - 阶段1：释放 comp 里多余的 block，并检查空闲 block 是否够（不够直接返回 None）；
   - 阶段2：处理前缀 token（含滑动窗口外释放、connector 外部 token 的 block）；
   - 阶段3：为 `new + lookahead`（含投机解码前瞻）分配新 block。
3. **分配失败 = 不调度**：`allocate_slots` 返回 `None` 时，Scheduler 会把该请求
   放回 waiting（或抢占 running 里的请求），而不是硬塞。
4. **引用计数回收**：`free()` 只做 ref_cnt -1，归零才真正回收到空闲队列，
   这样共享前缀的多个请求不会互相影响。
