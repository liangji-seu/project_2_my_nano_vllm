import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from my_vllm.config import EngineConfig
from my_vllm.attention.metadata import (
    FullAttentionMetadata,
    FullAttentionMetadataCollection,
)
from my_vllm.model_executor.model_loader import load_model
from my_vllm.model_executor.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.kv_cache_interface import FullAttentionSpec, generate_kv_cache_config
from my_vllm.v1.request import SamplingParams
from my_vllm.worker.gpu_model_runner import GPUModelRunner
from my_vllm.worker.gpu_input_batch import CachedRequestState


def tiny_hf_config(tie_word_embeddings=False):
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": True,
        "torch_dtype": "float32",
    }


def write_checkpoint(tmp_path, config_dict):
    (tmp_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config.from_dict(config_dict))
    safetensors_torch.save_file(model.state_dict(), tmp_path / "model.safetensors")
    return model


def test_full_attention_page_size_and_layout():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype="float16",
    )
    assert spec.page_size_bytes == 2 * 16 * 2 * 8 * 2
    config = generate_kv_cache_config(
        {"layer.0": spec, "layer.1": spec},
        available_memory=spec.page_size_bytes * 2 * 10,
    )
    assert config.num_blocks == 10
    assert len(config.kv_cache_tensors) == 2
    assert all(t.size == spec.page_size_bytes * 10 for t in config.kv_cache_tensors)


def test_delayed_pp_token_can_precede_multiple_synced_tokens(monkeypatch):
    """PP>2 时，延迟 token 已不一定是本地 output 的最后一项。"""

    monkeypatch.setattr(
        "my_vllm.worker.gpu_model_runner.is_pipeline_last_stage",
        lambda: False,
    )
    pending = SimpleNamespace(
        sampled_tokens=torch.tensor([10], dtype=torch.int64),
        req_ids=("request",),
        should_sample=(True,),
        output_lengths=(0,),
    )
    runner = object.__new__(GPUModelRunner)
    runner.pp_token_handler = SimpleNamespace(begin_step=lambda: pending)
    runner.input_batch = None
    runner.requests = {
        "request": CachedRequestState(
            req_id="request",
            prompt_token_ids=[1],
            output_token_ids=[10, 11, 12],
            sampling_params=SamplingParams(max_tokens=4),
            block_ids=([1],),
            num_computed_tokens=3,
        )
    }

    runner._consume_delayed_pp_tokens()

    assert runner.requests["request"].output_token_ids == [10, 11, 12]


def test_safetensors_recursively_matches_parameter_tree(tmp_path):
    config_dict = tiny_hf_config()
    source = write_checkpoint(tmp_path, config_dict)
    loaded, _, dtype, loaded_names = load_model(
        str(tmp_path), torch.device("cpu"), "float32", "safetensors"
    )
    assert dtype == torch.float32
    assert loaded_names == set(dict(loaded.named_parameters()))
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, dict(source.named_parameters())[name])


def test_profile_and_kv_cache_block_view_without_input_batch(tmp_path):
    config_dict = tiny_hf_config()
    write_checkpoint(tmp_path, config_dict)
    engine_config = EngineConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=16,
        max_num_batched_tokens=16,
        block_size=4,
    )
    runner = GPUModelRunner(engine_config, torch.device("cpu"))
    runner.load_model()
    runner.profile_run()
    specs = runner.get_kv_cache_spec()
    per_block = sum(spec.page_size_bytes for spec in specs.values())
    kv_config = generate_kv_cache_config(specs, available_memory=per_block * 8)
    runner.initialize_kv_cache(kv_config)

    assert set(runner.kv_caches) == set(specs)
    for layer_name, cache in runner.kv_caches.items():
        spec = specs[layer_name]
        assert cache.shape == (
            8,
            2,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        assert dict(runner.model.attention_layers())[layer_name].kv_cache is cache


def make_flat_attention_metadata(model, query_lens):
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start = [0]
    for query_len in query_lens:
        query_start.append(query_start[-1] + query_len)
    positions = torch.cat(
        [torch.arange(query_len) for query_len in query_lens]
    )
    metadata = FullAttentionMetadata(
        kv_cache_group_id=0,
        layer_names=tuple(name for name, _ in model.attention_layers()),
        block_size=4,
        causal=True,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens),
        query_start_loc=torch.tensor(query_start, dtype=torch.int32),
        seq_lens=torch.tensor(query_lens, dtype=torch.int32),
        num_computed_tokens=torch.zeros(num_reqs, dtype=torch.int32),
        num_scheduled_tokens=torch.tensor(query_lens, dtype=torch.int32),
        positions=positions,
        block_table=torch.zeros((num_reqs, 2), dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
    )
    return positions, FullAttentionMetadataCollection(
        by_group={0: metadata},
        by_layer={name: metadata for name in metadata.layer_names},
    )


def test_qwen2_flat_ragged_batch_matches_separate_prefills():
    torch.manual_seed(11)
    model = Qwen2ForCausalLM(Qwen2Config.from_dict(tiny_hf_config())).eval()
    first = torch.tensor([1, 2, 3], dtype=torch.int64)
    second = torch.tensor([4, 5], dtype=torch.int64)
    flat_input_ids = torch.cat((first, second))
    positions, metadata = make_flat_attention_metadata(model, [3, 2])

    flat_hidden = model(flat_input_ids, positions, metadata)
    expected = torch.cat((model(first), model(second)))

    assert flat_hidden.shape == (5, 16)
    torch.testing.assert_close(flat_hidden, expected, rtol=1e-5, atol=1e-6)
    assert model.compute_logits(flat_hidden).shape == (5, 32)


def test_preprocessed_flat_inputs_run_real_tiny_qwen2(tmp_path):
    config_dict = tiny_hf_config()
    write_checkpoint(tmp_path, config_dict)
    engine_config = EngineConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=16,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        block_size=4,
    )
    runner = GPUModelRunner(engine_config, torch.device("cpu"))
    runner.load_model()
    specs = runner.get_kv_cache_spec()
    per_block = sum(spec.page_size_bytes for spec in specs.values())
    runner.initialize_kv_cache(
        generate_kv_cache_config(specs, available_memory=per_block * 8)
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="request",
                prompt_token_ids=[1, 2, 3],
                output_token_ids=[],
                sampling_params=SamplingParams(max_tokens=2),
                block_ids=([1],),
                num_computed_tokens=0,
                max_tokens=2,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"request": 3},
        total_num_scheduled_tokens=3,
        finished_req_ids=set(),
    )

    runner.execute_model(scheduler_output)
    runner.sample_tokens()
    assert runner.last_model_inputs is not None
    hidden_states = runner.model_forward(runner.last_model_inputs)
    sample_hidden_states = hidden_states[
        runner.last_model_inputs.logits_indices
    ]
    logits = runner.model.compute_logits(sample_hidden_states)

    assert hidden_states.shape == (3, 16)
    assert sample_hidden_states.shape == (1, 16)
    assert logits.shape == (1, 32)


def test_kv_cache_prefill_and_decode_match_dense_recomputation(tmp_path):
    """真实 execute_model 的第二步必须从分页 cache 读到第一步全部 KV。"""

    config_dict = tiny_hf_config()
    write_checkpoint(tmp_path, config_dict)
    engine_config = EngineConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=16,
        max_num_seqs=2,
        max_num_batched_tokens=8,
        block_size=4,
    )
    runner = GPUModelRunner(engine_config, torch.device("cpu"))
    runner.load_model()
    specs = runner.get_kv_cache_spec()
    per_block = sum(spec.page_size_bytes for spec in specs.values())
    runner.initialize_kv_cache(
        generate_kv_cache_config(specs, available_memory=per_block * 8)
    )

    prompt = [1, 2, 3]
    with torch.inference_mode():
        dense_hidden = runner.model(torch.tensor(prompt, dtype=torch.int64))
        expected_first = int(
            runner.model.compute_logits(dense_hidden[-1:]).argmax(dim=-1).item()
        )
    first = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="request",
                prompt_token_ids=prompt,
                output_token_ids=[],
                sampling_params=SamplingParams(max_tokens=2),
                block_ids=([1],),
                num_computed_tokens=0,
                max_tokens=2,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"request": 3},
        total_num_scheduled_tokens=3,
        finished_req_ids=set(),
    )
    runner.execute_model(first)
    first_output = runner.sample_tokens()
    assert first_output.sampled_token_ids == [[expected_first]]

    full_ids = prompt + [expected_first]
    with torch.inference_mode():
        dense_hidden = runner.model(torch.tensor(full_ids, dtype=torch.int64))
        expected_second = int(
            runner.model.compute_logits(dense_hidden[-1:]).argmax(dim=-1).item()
        )
    second = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["request"],
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_scheduled_tokens=[1],
        ),
        num_scheduled_tokens={"request": 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
    )
    runner.execute_model(second)
    second_output = runner.sample_tokens()
    assert second_output.sampled_token_ids == [[expected_second]]
