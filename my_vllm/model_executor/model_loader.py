"""最小模型 registry 与 safetensors 流式加载器。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from my_vllm.model_executor.models import MODEL_REGISTRY
from my_vllm.model_executor.models.qwen2 import Qwen2Config

logger = logging.getLogger(__name__)


def resolve_torch_dtype(config_dtype: str, hf_config: dict) -> torch.dtype:
    name = config_dtype
    if name == "auto":
        name = hf_config.get("torch_dtype", "float16")
    aliases = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"不支持的 dtype: {name}")
    return aliases[name]


def read_hf_config(model_path: str) -> dict:
    path = Path(model_path).expanduser().resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到 Hugging Face config.json: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def initialize_model(hf_config: dict) -> nn.Module:
    architectures = hf_config.get("architectures") or []
    if len(architectures) != 1:
        raise ValueError(f"config.json 必须声明唯一 architecture，实际为 {architectures}")
    architecture = architectures[0]
    model_class = MODEL_REGISTRY.get(architecture)
    if model_class is None:
        raise ValueError(
            f"my-vLLM 暂不支持模型架构 {architecture}；"
            f"当前支持 {sorted(MODEL_REGISTRY)}"
        )
    if architecture == "Qwen2ForCausalLM":
        config = Qwen2Config.from_dict(hf_config)
    else:  # pragma: no cover - registry 扩展时必须同步增加 config 解析
        raise AssertionError(architecture)
    # meta device 只建立参数树/形状，不先在 CPU 或 GPU 分配整份随机权重。
    with torch.device("meta"):
        return model_class(config)


def safetensors_weights_iterator(
    model_path: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("加载 safetensors 需要安装 `pip install safetensors`") from exc

    folder = Path(model_path).expanduser().resolve()
    index_path = folder / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        files = sorted({folder / filename for filename in weight_map.values()})
    else:
        files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{folder} 中没有找到 *.safetensors 权重")

    for filename in files:
        # safe_open 在 CPU 上按需 mmap；get_tensor(name) 才触发对应页读取。
        with safe_open(str(filename), framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                yield name, checkpoint.get_tensor(name)


def _set_parameter(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    path, _, leaf = name.rpartition(".")
    module = model.get_submodule(path) if path else model
    parameter = module._parameters.get(leaf)
    if parameter is None:
        raise KeyError(name)
    if tuple(parameter.shape) != tuple(tensor.shape):
        raise ValueError(
            f"权重 shape 不匹配 {name}: model={tuple(parameter.shape)}, "
            f"checkpoint={tuple(tensor.shape)}"
        )
    target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
    module._parameters[leaf] = nn.Parameter(
        tensor.to(device=device, dtype=target_dtype),
        requires_grad=False,
    )


def load_model(
    model_path: str,
    device: torch.device,
    dtype_name: str,
    load_format: str,
) -> tuple[nn.Module, dict, torch.dtype, set[str]]:
    hf_config = read_hf_config(model_path)
    dtype = resolve_torch_dtype(dtype_name, hf_config)
    model = initialize_model(hf_config)

    expected = set(dict(model.named_parameters(remove_duplicate=True)))
    loaded: set[str] = set()
    if load_format == "dummy":
        for name, parameter in list(model.named_parameters(remove_duplicate=True)):
            tensor_dtype = dtype if parameter.is_floating_point() else parameter.dtype
            tensor = torch.empty(parameter.shape, device=device, dtype=tensor_dtype)
            nn.init.normal_(tensor, mean=0.0, std=0.02)
            _set_parameter(model, name, tensor, device, tensor_dtype)
        loaded = expected
    elif load_format == "safetensors":
        for name, tensor in safetensors_weights_iterator(model_path):
            # tied embedding checkpoint 可能带重复 lm_head；模型只保留共享参数一次。
            target_name = name
            if (
                name == "lm_head.weight"
                and name not in expected
                and "model.embed_tokens.weight" in expected
            ):
                target_name = "model.embed_tokens.weight"
            if target_name not in expected:
                logger.debug("跳过模型不使用的 checkpoint 权重 %s", name)
                continue
            if target_name in loaded:
                continue
            _set_parameter(model, target_name, tensor, device, dtype)
            loaded.add(target_name)
        missing = expected - loaded
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            raise RuntimeError(f"checkpoint 缺少 {len(missing)} 个模型参数，例如：{preview}")
    else:
        raise ValueError(f"不支持的 load_format: {load_format}")

    if getattr(model.config, "tie_word_embeddings", False):
        # 替换 parameter 对象后重新建立共享引用，不能只让两者数值相等。
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()
    return model, hf_config, dtype, loaded
