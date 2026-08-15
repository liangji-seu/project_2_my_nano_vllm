"""Scheduler 与分层 KV cache 账本测试（纯 Python，无需 GPU）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_vllm.config import EngineConfig
from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.output import ModelRunnerOutput
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from my_vllm.v1.request import FinishReason, Request, SamplingParams


def make_request(req_id, token_ids, max_tokens=1):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def mock_execute(scheduler, scheduler_output):
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled = [
        [ord("a") + scheduler.requests[req_id].num_output_tokens % 26]
        for req_id in req_ids
    ]
    return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)


def run_loop(scheduler, max_steps=100000):
    finished = {}
    for _ in range(max_steps):
        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens > 0:
            scheduler.update_from_output(
                scheduler_output,
                mock_execute(scheduler, scheduler_output),
            )
        for req_id in list(scheduler.finished_req_ids):
            request = scheduler.requests.get(req_id)
            scheduler.finished_req_ids.discard(req_id)
            if request is not None:
                finished[req_id] = request
                scheduler.requests.pop(req_id, None)
        if not scheduler.running and not scheduler.waiting:
            break
    return finished


def assert_no_leak(kv_cache_manager, num_gpu_blocks):
    pool = kv_cache_manager.block_pool
    leaked = [
        block
        for block in pool.blocks
        if not block.is_null and block.ref_cnt != 0
    ]
    assert not leaked
    assert pool.get_num_free_blocks() == num_gpu_blocks - 1


def test_block_pool_uses_shared_ids_and_o1_free_queue():
    pool = BlockPool(5, enable_caching=True, hash_block_size=16)
    assert pool.null_block.block_id == 0
    assert pool.get_num_free_blocks() == 4

    block1, block2 = pool.get_new_blocks(2)
    assert [block1.block_id, block2.block_id] == [1, 2]
    assert block1.prev_free_block is None
    assert block1.next_free_block is None

    pool.free_blocks(reversed([block1, block2]))
    assert pool.get_num_free_blocks() == 4


def test_request_owns_chained_content_hashes():
    first = make_request("first", list(range(8)))
    second = make_request("second", list(range(4)) + list(range(4, 8)))
    different_parent = make_request("third", [99, 98, 97, 96] + list(range(4, 8)))

    for request in (first, second, different_parent):
        request.update_block_hashes(4)

    assert first.block_hashes == second.block_hashes
    assert first.block_hashes[1] != different_parent.block_hashes[1]


def test_prefix_cache_is_registered_only_after_execution():
    kv = KVCacheManager(5, 16, max_model_len=64)
    source = make_request("source", list(range(32)))
    follower = make_request("follower", list(range(16)) + list(range(100, 116)))

    allocated = kv.allocate_slots(source, num_new_tokens=32)
    assert allocated is not None
    _, hit_before_execute = kv.get_computed_blocks(follower)
    assert hit_before_execute == 0

    kv.cache_blocks(source, num_computed_tokens=32)
    hit_blocks, hit_after_execute = kv.get_computed_blocks(follower)
    assert hit_after_execute == 16
    assert hit_blocks.get_block_ids()[0] == [1]

    kv.free(source)
    assert_no_leak(kv, 5)


def test_prefix_cache_shares_blocks_and_refcounts():
    kv = KVCacheManager(5, 16, max_model_len=64)
    source = make_request("source", list(range(32)))
    kv.allocate_slots(source, num_new_tokens=32)
    kv.cache_blocks(source, num_computed_tokens=32)

    follower = make_request(
        "follower",
        list(range(16)) + list(range(100, 116)),
    )
    hit_blocks, num_hit = kv.get_computed_blocks(follower)
    new_blocks = kv.allocate_slots(
        follower,
        num_new_tokens=16,
        num_new_computed_tokens=num_hit,
        new_computed_blocks=hit_blocks,
    )
    assert new_blocks is not None
    assert kv.get_block_ids("source")[0] == [1, 2]
    assert kv.get_block_ids("follower")[0] == [1, 3]
    assert kv.block_pool.blocks[1].ref_cnt == 2

    kv.free(source)
    assert kv.block_pool.blocks[1].ref_cnt == 1
    kv.free(follower)
    assert_no_leak(kv, 5)


def test_allocate_failure_does_not_leak_hit_refcount():
    kv = KVCacheManager(5, 16, max_model_len=64)
    source = make_request("source", list(range(32)))
    kv.allocate_slots(source, num_new_tokens=32)
    kv.cache_blocks(source, num_computed_tokens=32)

    follower = make_request(
        "follower",
        list(range(16)) + list(range(100, 148)),
    )
    hit_blocks, num_hit = kv.get_computed_blocks(follower)
    result = kv.allocate_slots(
        follower,
        num_new_tokens=48,
        num_new_computed_tokens=num_hit,
        new_computed_blocks=hit_blocks,
    )
    assert result is None
    assert kv.block_pool.blocks[1].ref_cnt == 1
    assert "follower" not in kv.coordinator.single_type_managers[0].req_to_blocks

    kv.free(source)
    hit_blocks, num_hit = kv.get_computed_blocks(follower)
    result = kv.allocate_slots(
        follower,
        num_new_tokens=48,
        num_new_computed_tokens=num_hit,
        new_computed_blocks=hit_blocks,
    )
    assert result is not None
    kv.free(follower)
    assert_no_leak(kv, 5)


def test_hybrid_coordinator_keeps_one_table_per_group():
    config = KVCacheConfig(
        num_blocks=13,
        kv_cache_tensors=[
            KVCacheTensor(size=4096, shared_by=["layer.0"]),
            KVCacheTensor(size=8192, shared_by=["layer.1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], FullAttentionSpec(block_size=4)),
            KVCacheGroupSpec(["layer.1"], FullAttentionSpec(block_size=8)),
        ],
    )
    kv = KVCacheManager(kv_cache_config=config, max_model_len=64)
    assert isinstance(kv.coordinator, HybridKVCacheCoordinator)
    assert kv.hash_block_size == 4
    assert kv.scheduler_block_size == 8

    source = make_request("source", list(range(16)))
    kv.allocate_slots(source, num_new_tokens=16)
    kv.cache_blocks(source, num_computed_tokens=16)
    assert [len(group) for group in kv.get_block_ids("source")] == [4, 2]

    follower = make_request("follower", list(range(16)) + list(range(100, 108)))
    hit_blocks, num_hit = kv.get_computed_blocks(follower)
    assert num_hit == 16
    assert [len(group) for group in hit_blocks.blocks] == [4, 2]

    new_blocks = kv.allocate_slots(
        follower,
        num_new_tokens=8,
        num_new_computed_tokens=num_hit,
        new_computed_blocks=hit_blocks,
    )
    assert new_blocks is not None
    assert [len(group) for group in kv.get_block_ids("follower")] == [6, 3]
    assert [len(group) for group in new_blocks.blocks] == [2, 1]

    kv.free(source)
    kv.free(follower)
    assert_no_leak(kv, 13)


def test_scheduler_output_preserves_all_group_block_tables():
    cache_config = KVCacheConfig(
        num_blocks=9,
        kv_cache_tensors=[KVCacheTensor(size=0), KVCacheTensor(size=0)],
        kv_cache_groups=[
            KVCacheGroupSpec([], FullAttentionSpec(block_size=4)),
            KVCacheGroupSpec([], FullAttentionSpec(block_size=8)),
        ],
    )
    kv = KVCacheManager(kv_cache_config=cache_config, max_model_len=32)
    scheduler = Scheduler(
        EngineConfig(max_model_len=32, max_num_batched_tokens=32),
        kv,
    )
    scheduler.add_request(make_request("request", list(range(16))))

    output = scheduler.schedule()
    request_data = output.scheduled_new_reqs[0]
    assert [len(group) for group in request_data.block_ids] == [4, 2]

    scheduler.update_from_output(output, mock_execute(scheduler, output))
    assert_no_leak(kv, 9)


def test_scheduler_full_loop_and_prefix_lifecycle():
    config = EngineConfig(
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=16,
        max_num_seqs=4,
        max_num_batched_tokens=64,
    )
    kv = KVCacheManager(
        config.num_gpu_blocks,
        config.block_size,
        config.max_model_len,
        config.enable_prefix_caching,
    )
    scheduler = Scheduler(config, kv)
    scheduler.add_request(make_request("A", list(range(32)), max_tokens=4))
    scheduler.add_request(
        make_request("B", list(range(16)) + list(range(32, 48)), max_tokens=3)
    )

    finished = run_loop(scheduler)
    assert set(finished) == {"A", "B"}
    assert finished["A"].get_finished_reason() == FinishReason.LENGTH
    assert finished["A"].num_output_tokens == 4
    assert finished["B"].num_output_tokens == 3
    assert_no_leak(kv, config.num_gpu_blocks)


def test_scheduler_preemption_has_no_refcount_leak():
    config = EngineConfig(
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=5,
        max_num_seqs=4,
        max_num_batched_tokens=64,
    )
    kv = KVCacheManager(
        config.num_gpu_blocks,
        config.block_size,
        config.max_model_len,
        config.enable_prefix_caching,
    )
    scheduler = Scheduler(config, kv)
    scheduler.add_request(make_request("A", list(range(16)), max_tokens=32))
    scheduler.add_request(make_request("B", list(range(100, 116)), max_tokens=32))

    finished = run_loop(scheduler)
    assert set(finished) == {"A", "B"}
    assert finished["B"].num_preemptions >= 1
    assert_no_leak(kv, config.num_gpu_blocks)
