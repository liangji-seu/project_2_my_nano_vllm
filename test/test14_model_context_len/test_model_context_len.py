import json
from argparse import Namespace

import pytest

from my_vllm.config import EngineConfig, make_arg_parser


def make_args(model, max_model_len=None):
    return Namespace(
        host="127.0.0.1",
        port=8010,
        model=str(model),
        disable_log_stats=False,
        served_model_name=None,
        max_model_len=max_model_len,
    )


def write_config(path, value):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(value))


def test_auto_resolves_model_context_limit(tmp_path):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": 32768})
    config = EngineConfig.from_cli_args(make_args(model))
    assert config.max_model_len == 32768
    assert config.model_context_limit == 32768
    assert config.max_model_len_source == "auto"


def test_cli_parser_defaults_to_model_context_limit(tmp_path):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": 32768})
    args = make_arg_parser().parse_args(["--model", str(model)])
    config = EngineConfig.from_cli_args(args)
    assert args.max_model_len is None
    assert config.max_model_len == 32768
    assert config.max_model_len_source == "auto"


def test_explicit_shorter_context_override(tmp_path):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": 32768})
    config = EngineConfig.from_cli_args(make_args(model, 1024))
    assert config.max_model_len == 1024
    assert config.model_context_limit == 32768
    assert config.max_model_len_source == "explicit"


def test_explicit_override_above_model_limit_is_rejected(tmp_path):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": 32768})
    with pytest.raises(ValueError, match="exceeds model"):
        EngineConfig.from_cli_args(make_args(model, 32769))


@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_explicit_override_is_rejected(tmp_path, value):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": 32768})
    with pytest.raises(ValueError, match="positive integer"):
        EngineConfig.from_cli_args(make_args(model, value))


def test_auto_without_local_model_config_fails_but_explicit_is_allowed(tmp_path):
    model = tmp_path / "missing"
    with pytest.raises(ValueError, match="specify --max-model-len explicitly"):
        EngineConfig.from_cli_args(make_args(model))
    config = EngineConfig.from_cli_args(make_args(model, 1024))
    assert config.max_model_len == 1024
    assert config.model_context_limit is None


def test_auto_requires_context_field_and_bad_config_is_rejected(tmp_path):
    model = tmp_path / "model"
    write_config(model, {"model_type": "unknown"})
    with pytest.raises(ValueError, match="max_position_embeddings"):
        EngineConfig.from_cli_args(make_args(model))
    (model / "config.json").write_text("{invalid")
    with pytest.raises(ValueError, match="cannot read model config"):
        EngineConfig.from_cli_args(make_args(model, 1024))


@pytest.mark.parametrize("value", ["32768", 0, True])
def test_bad_model_context_field_is_rejected_even_with_override(tmp_path, value):
    model = tmp_path / "model"
    write_config(model, {"max_position_embeddings": value})
    with pytest.raises(ValueError, match="max_position_embeddings must be a positive integer"):
        EngineConfig.from_cli_args(make_args(model, 1024))
