"""my-vLLM 支持的模型实现。"""

from my_vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

MODEL_REGISTRY = {
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
}

__all__ = ["MODEL_REGISTRY", "Qwen2ForCausalLM"]
