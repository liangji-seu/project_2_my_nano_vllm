import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from my_vllm.config import EngineConfig
from my_vllm.model_executor.model_loader import load_model
from my_vllm.model_executor.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from my_vllm.v1.kv_cache_interface import FullAttentionSpec, generate_kv_cache_config
from my_vllm.worker.gpu_model_runner import GPUModelRunner


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
