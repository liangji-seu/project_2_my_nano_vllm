"""Worker 内的模型构造、权重加载、profiling 与 KV Cache 物理初始化。"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from my_vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)


class GPUModelRunner:
    """持有一个 Worker rank 的模型和真实 KV Cache 张量。

    本阶段刻意不接 ``SchedulerOutput -> InputBatch``：真实模型只用于 profiling，
    在线 execute_model 仍保留确定性 mock 采样。
    """

    def __init__(self, vllm_config, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        self.model: nn.Module | None = None
        self.hf_config: dict = {}
        self.model_dtype = torch.float16
        self.model_memory_usage = 0
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.kv_cache_config: KVCacheConfig | None = None
        self._req_output_counts: dict[str, int] = {}

    @property
    def is_mock_model(self) -> bool:
        return self.vllm_config.model == "test-model"

    def load_model(self) -> None:
        if self.is_mock_model:
            logger.info("test-model 使用 mock 执行路径，跳过真实模型权重加载")
            return
        if self.vllm_config.parallel_config.world_size != 1:
            raise NotImplementedError(
                "真实 Qwen2 权重目前只支持 TP=1、PP=1；TP/PP 权重切分属于后续分布式阶段"
            )

        from my_vllm.model_executor.model_loader import load_model

        before = (
            torch.cuda.memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0
        )
        model, hf_config, dtype, loaded = load_model(
            self.vllm_config.model,
            self.device,
            self.vllm_config.dtype,
            self.vllm_config.load_format,
        )
        self.model = model
        self.hf_config = hf_config
        self.model_dtype = dtype
        if self.device.type == "cuda":
            self.model_memory_usage = torch.cuda.memory_allocated(self.device) - before
        else:
            self.model_memory_usage = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
        logger.info(
            "模型加载完成：architecture=%s, parameters=%d, loaded_tensors=%d, memory=%.2f GiB",
            hf_config.get("architectures", ["unknown"])[0],
            sum(p.numel() for p in model.parameters()),
            len(loaded),
            self.model_memory_usage / 1024**3,
        )

    def get_kv_cache_spec(self) -> dict[str, FullAttentionSpec]:
        if self.is_mock_model:
            return {
                "mock.layers.0.self_attn": FullAttentionSpec(
                    block_size=self.vllm_config.block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype="float16",
                )
            }
        assert self.model is not None
        dtype_name = str(self.model_dtype).removeprefix("torch.")
        if self.vllm_config.kv_cache_dtype != "auto":
            dtype_name = self.vllm_config.kv_cache_dtype
        return {
            name: FullAttentionSpec(
                block_size=self.vllm_config.block_size,
                num_kv_heads=attention.num_kv_heads,
                head_size=attention.head_dim,
                dtype=dtype_name,
            )
            for name, attention in self.model.attention_layers()
        }

    @torch.inference_mode()
    def profile_run(self) -> None:
        """用最大 token 预算跑一次无 KV Cache 的 dummy forward。"""

        if self.is_mock_model:
            if self.device.type == "cuda":
                # mock 模式也真正触发一次很小的 CUDA allocation，验证 profiling 链路。
                torch.empty(1, device=self.device)
                torch.cuda.synchronize(self.device)
            return
        assert self.model is not None
        num_tokens = min(
            self.vllm_config.max_num_batched_tokens,
            self.vllm_config.max_model_len,
            int(
                self.hf_config.get(
                    "max_position_embeddings", self.vllm_config.max_model_len
                )
            ),
        )
        vocab_size = int(self.hf_config["vocab_size"])
        input_ids = torch.randint(0, vocab_size, (1, num_tokens), device=self.device)
        hidden_states = self.model(input_ids)
        # vLLM 只对需要采样的位置算 logits。这里取最多 max_num_seqs 个末尾
        # hidden state，覆盖 logits/sampler 峰值但不引入 InputBatch 状态。
        num_sampled_positions = min(self.vllm_config.max_num_seqs, num_tokens)
        logits = self.model.compute_logits(hidden_states[:, -num_sampled_positions:, :])
        _ = torch.argmax(logits, dim=-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        del input_ids, hidden_states, logits

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """按 KVCacheTensor 分配 byte buffer，再 reshape 出 block 第一维。"""

        specs_by_layer = {
            layer_name: group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        raw_by_layer: dict[str, torch.Tensor] = {}
        self.kv_caches = {}
        for tensor_config in kv_cache_config.kv_cache_tensors:
            raw = torch.zeros(tensor_config.size, dtype=torch.uint8, device=self.device)
            for layer_name in tensor_config.shared_by:
                raw_by_layer[layer_name] = raw

        dtype_map = {
            "float16": torch.float16,
            "torch.float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "torch.float32": torch.float32,
        }
        for layer_name, spec in specs_by_layer.items():
            if not isinstance(spec, FullAttentionSpec):
                raise NotImplementedError(type(spec).__name__)
            raw = raw_by_layer[layer_name]
            cache = raw.view(dtype_map[spec.dtype]).view(
                kv_cache_config.num_blocks,
                2,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
            )
            self.kv_caches[layer_name] = cache

        if self.model is not None:
            attention_layers = dict(self.model.attention_layers())
            for layer_name, cache in self.kv_caches.items():
                attention_layers[layer_name].kv_cache = cache
        self.kv_cache_config = kv_cache_config
        logger.info(
            "KV Cache 物理初始化完成：layers=%d, blocks=%d, bytes=%.2f GiB",
            len(self.kv_caches),
            kv_cache_config.num_blocks,
            sum(t.size for t in kv_cache_config.kv_cache_tensors) / 1024**3,
        )

    def get_model(self) -> nn.Module | None:
        return self.model

    def execute_model(self, scheduler_output):
        """InputBatch 尚未实现，继续使用可验证控制面的确定性 mock 采样。"""
        from my_vllm.v1.core.sched.output import ModelRunnerOutput

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        sampled_token_ids = []
        for req_id in req_ids:
            count = self._req_output_counts.get(req_id, 0)
            sampled_token_ids.append([ord("a") + count % 26])
            self._req_output_counts[req_id] = count + 1
        for req_id in scheduler_output.finished_req_ids:
            self._req_output_counts.pop(req_id, None)
        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )
