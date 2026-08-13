"""
调度器 + KV cache 单元测试（纯 Python，无需 torch / GPU）

覆盖：
  1. BlockPool        — 分配 / 释放 / 引用计数 / 驱逐顺序
  2. KVCacheManager   — 前缀缓存命中共享 + 引用计数
  3. KVCacheManager   — 显存不足分配失败不泄漏引用计数（关键 bug 回归）
  4. Scheduler        — 完整 schedule → execute → update 主循环，请求正常结束
  5. Scheduler        — 抢占 + 恢复，无引用计数泄漏

运行：
  /path/to/vllm_study/bin/python test/test7_scheduler_kv_cache/test_scheduler_kv_cache.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_vllm.config import EngineConfig
from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.kv_cache_utils import BlockPool
from my_vllm.v1.core.sched.output import ModelRunnerOutput
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.request import FinishReason, Request, RequestStatus, SamplingParams


def make_request(req_id, token_ids, max_tokens):
    """构造一个测试请求"""
    return Request(
        request_id=req_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def mock_execute(scheduler, scheduler_output):
    """模拟一次前向 + 采样：每个请求每步生成一个确定性 token

    与 EngineCore._execute_model() 的占位逻辑一致：按已生成长度轮转 a..z。
    """
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    sampled = []
    for rid in req_ids:
        req = scheduler.requests[rid]
        sampled.append([ord("a") + (req.num_output_tokens % 26)])
    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=sampled,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
    )


def run_loop(scheduler, max_steps=100000):
    """模拟 EngineCore 主循环：schedule → execute → update → 收集结束请求

    返回 {request_id: 已结束的 Request}。
    """
    finished = {}
    for _ in range(max_steps):
        out = scheduler.schedule()
        if out.total_num_scheduled_tokens > 0:
            runner_out = mock_execute(scheduler, out)
            scheduler.update_from_output(out, runner_out)
        for rid in list(scheduler.finished_req_ids):
            req = scheduler.requests.get(rid)
            scheduler.finished_req_ids.discard(rid)
            if req is not None:
                finished[rid] = req
                scheduler.requests.pop(rid, None)
        if not scheduler.running and not scheduler.waiting and not scheduler.finished_req_ids:
            break
    return finished


def assert_no_leak(kv_cache_manager, num_gpu_blocks):
    """所有 block 引用计数归零、空闲队列恢复到满"""
    pool = kv_cache_manager.block_pool
    leaked = [b for b in pool.blocks if b.ref_cnt != 0]
    assert not leaked, f"引用计数泄漏的 block: {leaked}"
    assert pool.get_num_free_blocks() == num_gpu_blocks, (
        f"空闲 block 数 {pool.get_num_free_blocks()} != {num_gpu_blocks}"
    )


# ==================================================================
# 1. BlockPool 基础
# ==================================================================

def test_block_pool():
    pool = BlockPool(4, enable_caching=True)
    assert pool.get_num_free_blocks() == 4

    b0, b1 = pool.get_new_blocks(2)
    assert [b0.block_id, b1.block_id] == [0, 1]
    assert pool.get_num_free_blocks() == 2
    assert b0.ref_cnt == 1 and b1.ref_cnt == 1

    pool.free_blocks([b0, b1])  # 按分配顺序释放
    assert pool.get_num_free_blocks() == 4
    assert b0.ref_cnt == 0 and b1.ref_cnt == 0


# ==================================================================
# 2. 前缀缓存命中共享 + 引用计数
# ==================================================================

def test_prefix_cache_refcount():
    # block_size=16：32 个 token 正好 2 个 block
    kv = KVCacheManager(num_gpu_blocks=4, block_size=16, max_model_len=64)

    a = make_request("A", list(range(32)), max_tokens=1)   # block0: 0..15, block1: 16..31
    kv.allocate_slots(a, num_new_tokens=32)
    assert kv.get_block_ids("A")[0] == [0, 1]

    # B 的前 16 个 token 与 A 相同 → 命中前缀缓存 block0
    b = make_request("B", list(range(16)) + list(range(32, 48)), max_tokens=1)
    hit_blocks, num_hit = kv.get_computed_blocks(b)
    assert num_hit == 16
    assert hit_blocks.blocks[0][0].block_id == 0

    kv.allocate_slots(b, num_new_tokens=16, num_new_computed_tokens=16,
                      new_computed_blocks=hit_blocks)
    assert kv.get_block_ids("B")[0] == [0, 2]  # 共享 block0 + 新 block2

    # block0 被 A、B 共享 → ref_cnt = 2
    pool = kv.block_pool
    assert pool.blocks[0].ref_cnt == 2

    # 释放 A：block0 ref_cnt 2→1（仍被 B 持有），block1 归零
    kv.free(a)
    assert pool.blocks[0].ref_cnt == 1
    assert pool.blocks[1].ref_cnt == 0

    # 释放 B：block0、block2 都归零
    kv.free(b)
    assert_no_leak(kv, 4)


# ==================================================================
# 3. 分配失败不泄漏引用计数（回归测试）
# ==================================================================

def test_allocate_fail_no_leak():
    kv = KVCacheManager(num_gpu_blocks=4, block_size=16, max_model_len=64)

    # A 占满 2 个 block（0, 1），free 剩 2
    a = make_request("A", list(range(32)), max_tokens=1)
    kv.allocate_slots(a, num_new_tokens=32)

    # B：16 共享 + 48 新 token = 64 token，需要 4 个 block，但只有 3 个可用
    b = make_request("B", list(range(16)) + list(range(100, 148)), max_tokens=1)
    hit_blocks, num_hit = kv.get_computed_blocks(b)
    assert num_hit == 16

    result = kv.allocate_slots(b, num_new_tokens=48, num_new_computed_tokens=16,
                               new_computed_blocks=hit_blocks)
    assert result is None  # 显存不足，分配失败

    # 关键断言：失败时命中的 block0 引用计数不能被白加一次
    assert kv.block_pool.blocks[0].ref_cnt == 1, (
        f"分配失败后 block0 ref_cnt 应为 1，实际 {kv.block_pool.blocks[0].ref_cnt}"
    )
    assert kv.block_pool.get_num_free_blocks() == 2

    # 释放 A 后重试，B 应能成功分配（前缀命中 block0 + 3 个新 block）
    kv.free(a)
    hit_blocks, num_hit = kv.get_computed_blocks(b)
    result = kv.allocate_slots(b, num_new_tokens=48, num_new_computed_tokens=16,
                               new_computed_blocks=hit_blocks)
    assert result is not None
    assert len(kv.get_block_ids("B")[0]) == 4

    kv.free(b)
    assert_no_leak(kv, 4)


# ==================================================================
# 4. 完整调度主循环：两个请求正常结束
# ==================================================================

def test_scheduler_full_loop():
    cfg = EngineConfig(max_model_len=64, block_size=16, num_gpu_blocks=16,
                       max_num_seqs=4, max_num_batched_tokens=64)
    kv = KVCacheManager(cfg.num_gpu_blocks, cfg.block_size,
                        cfg.max_model_len, cfg.enable_prefix_caching)
    sched = Scheduler(cfg, kv)

    # A：32 token prompt；B：前 16 与 A 共享 + 16 新 token
    sched.add_request(make_request("A", list(range(32)), max_tokens=4))
    sched.add_request(make_request("B", list(range(16)) + list(range(32, 48)),
                                   max_tokens=3))

    finished = run_loop(sched)

    assert set(finished) == {"A", "B"}
    assert finished["A"].get_finished_reason() == FinishReason.LENGTH
    assert finished["B"].get_finished_reason() == FinishReason.LENGTH
    assert finished["A"].num_output_tokens == 4
    assert finished["B"].num_output_tokens == 3
    assert_no_leak(kv, cfg.num_gpu_blocks)


# ==================================================================
# 5. 抢占 + 恢复：无引用计数泄漏
# ==================================================================

def test_scheduler_preemption():
    # 4 个 block 只够 A、B 各长到 2 个 block；第 3 个 block 会触发抢占
    cfg = EngineConfig(max_model_len=64, block_size=16, num_gpu_blocks=4,
                       max_num_seqs=4, max_num_batched_tokens=64)
    kv = KVCacheManager(cfg.num_gpu_blocks, cfg.block_size,
                        cfg.max_model_len, cfg.enable_prefix_caching)
    sched = Scheduler(cfg, kv)

    # A、B 各 16 token prompt，各要生成 32 token（总长 48 = 3 block），互相无前缀重叠
    sched.add_request(make_request("A", list(range(16)), max_tokens=32))
    sched.add_request(make_request("B", list(range(100, 116)), max_tokens=32))

    finished = run_loop(sched)

    assert set(finished) == {"A", "B"}
    assert finished["B"].num_preemptions >= 1, (
        f"B 应至少被抢占一次，实际 {finished['B'].num_preemptions}"
    )
    assert finished["A"].num_output_tokens == 32
    assert finished["B"].num_output_tokens == 32
    assert_no_leak(kv, cfg.num_gpu_blocks)


# ==================================================================
# 运行器
# ==================================================================

def main():
    tests = [
        test_block_pool,
        test_prefix_cache_refcount,
        test_allocate_fail_no_leak,
        test_scheduler_full_loop,
        test_scheduler_preemption,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
        except Exception:
            print(f"[FAIL] {test.__name__} 异常:")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} 通过")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
