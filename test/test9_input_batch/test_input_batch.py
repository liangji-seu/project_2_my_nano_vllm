from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.request import SamplingParams
from my_vllm.worker.gpu_input_batch import CachedRequestState, InputBatch


def make_state(
    req_id: str,
    prompt: list[int],
    output: list[int],
    block_ids: tuple[list[int], ...],
    computed: int,
) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        output_token_ids=output,
        sampling_params=SamplingParams(max_tokens=8),
        block_ids=block_ids,
        num_computed_tokens=computed,
    )


def test_input_batch_condense_moves_all_persistent_state():
    batch = InputBatch(max_num_reqs=4, max_model_len=16, num_kv_cache_groups=2)
    batch.add_request(make_state("A", [1, 2], [3], ([10], [20]), 2))
    batch.add_request(make_state("B", [4], [], ([11], [21]), 0))
    batch.add_request(make_state("C", [5, 6], [7, 8], ([12], [22]), 3))

    assert batch.remove_request("B") == 1
    batch.condense()

    assert batch.req_ids == ["A", "C"]
    assert batch.req_id_to_index == {"A": 0, "C": 1}
    assert batch.snapshot("C") == {
        "req_index": 1,
        "token_ids": [5, 6, 7, 8],
        "num_prompt_tokens": 2,
        "num_tokens": 4,
        "num_computed_tokens": 3,
        "block_ids": ([12], [22]),
    }


def make_new(
    req_id: str,
    prompt: list[int],
    *,
    output: list[int] | None = None,
    computed: int = 0,
    blocks: list[int] | None = None,
) -> NewRequestData:
    sampling = SamplingParams(max_tokens=8)
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        output_token_ids=list(output or []),
        sampling_params=sampling,
        block_ids=(list(blocks or [1]),),
        num_computed_tokens=computed,
        max_tokens=sampling.max_tokens,
    )


def make_output(
    *,
    new: list[NewRequestData] | None = None,
    cached: CachedRequestData | None = None,
    scheduled: dict[str, int],
    finished: set[str] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=list(new or []),
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        finished_req_ids=set(finished or set()),
    )


def test_model_runner_updates_persistent_batch_and_bookkeeping():
    torch = __import__("torch")
    from my_vllm.config import EngineConfig
    from my_vllm.v1.kv_cache_interface import make_default_kv_cache_config
    from my_vllm.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(
        EngineConfig(
            model="test-model",
            max_model_len=16,
            max_num_seqs=4,
            block_size=4,
        ),
        torch.device("cpu"),
    )
    runner.initialize_kv_cache(
        make_default_kv_cache_config(num_blocks=16, block_size=4)
    )

    # A/B 首次进入；本轮都走到已知序列末尾，所以各自产生第一个 mock token a。
    first = make_output(
        new=[make_new("A", [10, 11]), make_new("B", [20])],
        scheduled={"A": 2, "B": 1},
    )
    first_result = runner.execute_model(first)
    assert first_result.sampled_token_ids == [[ord("a")], [ord("a")]]
    assert runner.input_batch.req_ids == ["A", "B"]
    assert runner.input_batch.snapshot("A")["token_ids"] == [10, 11, ord("a")]
    assert runner.requests["A"].num_computed_tokens == 2

    # B 已完成被移除；A 只收 computed/new block 增量并生成 b。
    second = make_output(
        cached=CachedRequestData(
            req_ids=["A"],
            new_block_ids=[([2],)],
            num_computed_tokens=[2],
            num_scheduled_tokens=[1],
        ),
        scheduled={"A": 1},
        finished={"B"},
    )
    second_result = runner.execute_model(second)
    assert second_result.sampled_token_ids == [[ord("b")]]
    assert set(runner.requests) == {"A"}
    assert runner.input_batch.req_ids == ["A"]
    assert runner.input_batch.snapshot("A") == {
        "req_index": 0,
        "token_ids": [10, 11, ord("a"), ord("b")],
        "num_prompt_tokens": 2,
        "num_tokens": 4,
        "num_computed_tokens": 3,
        "block_ids": ([1, 2],),
    }


def test_chunked_prefill_samples_only_on_last_chunk_and_resume_keeps_outputs():
    torch = __import__("torch")
    from my_vllm.config import EngineConfig
    from my_vllm.v1.kv_cache_interface import make_default_kv_cache_config
    from my_vllm.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(
        EngineConfig(model="test-model", max_model_len=16, block_size=4),
        torch.device("cpu"),
    )
    runner.initialize_kv_cache(
        make_default_kv_cache_config(num_blocks=16, block_size=4)
    )

    first = make_output(new=[make_new("C", [1, 2, 3, 4])], scheduled={"C": 2})
    assert runner.execute_model(first).sampled_token_ids == [[]]
    assert runner.requests["C"].num_computed_tokens == 2

    second = make_output(
        cached=CachedRequestData(
            req_ids=["C"],
            new_block_ids=[([],)],
            num_computed_tokens=[2],
            num_scheduled_tokens=[2],
        ),
        scheduled={"C": 2},
    )
    assert runner.execute_model(second).sampled_token_ids == [[ord("a")]]

    # 模拟抢占后作为完整 NewRequestData 恢复；output token a 不能丢失。
    resumed = make_output(
        new=[
            make_new(
                "C",
                [1, 2, 3, 4],
                output=[ord("a")],
                computed=0,
                blocks=[7, 8],
            )
        ],
        scheduled={"C": 5},
    )
    assert runner.execute_model(resumed).sampled_token_ids == [[ord("b")]]
    assert runner.requests["C"].all_token_ids == [1, 2, 3, 4, ord("a"), ord("b")]
    assert runner.input_batch.snapshot("C")["block_ids"] == ([7, 8],)


def test_scheduler_runner_loop_cleans_finished_worker_state():
    torch = __import__("torch")
    from my_vllm.config import EngineConfig
    from my_vllm.v1.core.kv_cache_manager import KVCacheManager
    from my_vllm.v1.core.sched.scheduler import Scheduler
    from my_vllm.v1.kv_cache_interface import make_default_kv_cache_config
    from my_vllm.v1.request import Request
    from my_vllm.worker.gpu_model_runner import GPUModelRunner

    config = EngineConfig(
        model="test-model",
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=5,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    kv_manager = KVCacheManager(
        config.num_gpu_blocks,
        config.block_size,
        config.max_model_len,
        config.enable_prefix_caching,
    )
    scheduler = Scheduler(config, kv_manager)
    runner = GPUModelRunner(config, torch.device("cpu"))
    runner.initialize_kv_cache(
        make_default_kv_cache_config(
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
    )
    scheduler.add_request(
        Request("A", list(range(16)), SamplingParams(max_tokens=4))
    )
    scheduler.add_request(
        Request("B", list(range(100, 116)), SamplingParams(max_tokens=4))
    )

    finished = {}
    for _ in range(100):
        scheduler_output = scheduler.schedule()
        if (
            scheduler_output.total_num_scheduled_tokens
            or scheduler_output.finished_req_ids
        ):
            model_output = runner.execute_model(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens:
                scheduler.update_from_output(scheduler_output, model_output)

        for req_id in list(scheduler.finished_req_ids):
            request = scheduler.requests.pop(req_id, None)
            scheduler.finished_req_ids.discard(req_id)
            if request is not None:
                finished[req_id] = request

        if (
            not scheduler.running
            and not scheduler.waiting
            and not scheduler._finished_req_ids_to_notify
        ):
            break

    assert set(finished) == {"A", "B"}
    assert finished["A"].output_token_ids == [ord("a"), ord("b"), ord("c"), ord("d")]
    assert finished["B"].output_token_ids == [ord("a"), ord("b"), ord("c"), ord("d")]
    assert runner.requests == {}
    assert runner.input_batch.req_ids == []
    assert kv_manager.block_pool.get_num_free_blocks() == config.num_gpu_blocks - 1


def test_prepare_inputs_flattens_batch_and_computes_slot_mapping():
    torch = __import__("torch")
    from my_vllm.config import EngineConfig
    from my_vllm.v1.kv_cache_interface import make_default_kv_cache_config
    from my_vllm.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(
        EngineConfig(
            model="test-model",
            max_model_len=16,
            max_num_seqs=4,
            max_num_batched_tokens=8,
            block_size=4,
        ),
        torch.device("cpu"),
    )
    runner.initialize_kv_cache(
        make_default_kv_cache_config(num_blocks=32, block_size=4)
    )

    first = make_output(
        new=[
            make_new("A", [1, 2, 3, 4], blocks=[3]),
            make_new("B", [10, 11, 12, 13, 14], computed=4, blocks=[5, 6]),
        ],
        scheduled={"A": 2, "B": 1},
    )
    runner.execute_model(first)
    prepared = runner.last_prepared_inputs
    assert prepared is not None
    assert prepared.num_reqs == 2
    assert prepared.num_tokens == 3
    assert prepared.input_ids.tolist() == [1, 2, 14]
    assert prepared.positions.tolist() == [0, 1, 4]
    assert prepared.query_start_loc.tolist() == [0, 2, 3]
    assert prepared.seq_lens.tolist() == [2, 5]
    assert prepared.slot_mappings[0].tolist() == [12, 13, 24]
    assert runner.req_indices.gpu[:3].tolist() == [0, 0, 1]
    assert runner.query_pos.gpu[:3].tolist() == [0, 1, 0]
    assert runner.token_indices.gpu[:3].tolist() == [0, 1, 20]
    assert runner.num_scheduled_tokens.gpu[:2].tolist() == [2, 1]
    assert runner.num_computed_tokens[:2].tolist() == [0, 4]
    assert runner.prev_positions.gpu[:2].tolist() == [-1, -1]
    assert runner.prev_num_draft_tokens.gpu[:2].tolist() == [0, 0]
    assert runner.num_decode_draft_tokens.gpu[:2].tolist() == [-1, -1]
    assert runner.num_accepted_tokens.gpu[:2].tolist() == [1, 1]
    assert runner.is_token_ids.gpu[:3].tolist() == [True, True, True]
    assert runner.inputs_embeds is None

    # A 离开 batch 后 B 从槽位 1 移到 0；prev_positions 保留上一批位置 1。
    second = make_output(
        cached=CachedRequestData(
            req_ids=["B"],
            new_block_ids=[([],)],
            num_computed_tokens=[5],
            num_scheduled_tokens=[1],
        ),
        scheduled={"B": 1},
    )
    runner.execute_model(second)
    prepared = runner.last_prepared_inputs
    assert prepared is not None
    assert prepared.input_ids.tolist() == [ord("a")]
    assert prepared.positions.tolist() == [5]
    assert prepared.query_start_loc.tolist() == [0, 1]
    assert prepared.seq_lens.tolist() == [6]
    assert prepared.slot_mappings[0].tolist() == [25]
    assert runner.token_indices.gpu[:1].tolist() == [5]
    assert runner.prev_positions.gpu[:1].tolist() == [1]


def test_prepare_inputs_uses_each_kv_group_block_size_for_slots():
    torch = __import__("torch")
    from my_vllm.config import EngineConfig
    from my_vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from my_vllm.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(
        EngineConfig(
            model="test-model",
            max_model_len=16,
            max_num_seqs=2,
            max_num_batched_tokens=4,
        ),
        torch.device("cpu"),
    )
    runner.initialize_kv_cache(
        KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[KVCacheTensor(size=0), KVCacheTensor(size=0)],
            kv_cache_groups=[
                KVCacheGroupSpec([], FullAttentionSpec(block_size=2)),
                KVCacheGroupSpec([], FullAttentionSpec(block_size=4)),
            ],
        )
    )
    request = make_new("A", [1, 2, 3, 4, 5], computed=3)
    request.block_ids = ([7, 8, 9], [4, 5])
    runner.execute_model(make_output(new=[request], scheduled={"A": 2}))

    prepared = runner.last_prepared_inputs
    assert prepared is not None
    assert prepared.positions.tolist() == [3, 4]
    assert prepared.slot_mappings[0].tolist() == [17, 18]
    assert prepared.slot_mappings[1].tolist() == [19, 20]
