"""Worker 内的模型构造、权重加载、profiling 与 KV Cache 物理初始化。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from my_vllm.attention.metadata import (
    FullAttentionMetadata,
    FullAttentionMetadataCollection,
)
from my_vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from my_vllm.worker.cpu_gpu_buffer import CpuGpuBuffer
from my_vllm.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = logging.getLogger(__name__)


class PreparedInputBuffers:
    """本轮已经整理好的模型输入视图，不包含 Attention metadata。"""

    def __init__(
        self,
        *,
        num_reqs: int,
        num_tokens: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mappings: dict[int, torch.Tensor],
    ) -> None:
        self.num_reqs = num_reqs
        self.num_tokens = num_tokens
        self.input_ids = input_ids
        self.positions = positions
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens
        self.slot_mappings = slot_mappings


@dataclass(frozen=True)
class ModelForwardInputs:
    """纯文本 decoder 模型一次前向真正消费的输入。"""

    input_ids: torch.Tensor
    inputs_embeds: None
    positions: torch.Tensor
    attention_metadata: FullAttentionMetadataCollection
    logits_indices: torch.Tensor


class GPUModelRunner:
    """持有一个 Worker rank 的模型和真实 KV Cache 张量。

    接通 ``SchedulerOutput -> InputBatch -> model -> sampler -> KV cache`` 的
    单卡同步 baseline。
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
        self.requests: dict[str, CachedRequestState] = {}
        # KV cache group 数要到 initialize_kv_cache 才能确定，因此延迟构造。
        self.input_batch: InputBatch | None = None
        self.max_num_tokens = vllm_config.max_num_batched_tokens
        self.max_num_reqs = vllm_config.max_num_seqs

        # 每个 step 都会覆写的固定地址工作缓冲。InputBatch 保存跨 step 状态，
        # 这些字段只描述当前 SchedulerOutput 的扁平化快照。
        self.input_ids = self._make_buffer(self.max_num_tokens, torch.int32)
        self.is_token_ids = self._make_buffer(
            self.max_num_tokens, torch.bool, fill_value=True
        )
        # 当前请求协议不接受 prompt embeds；后续接入时再按 hidden_size 分配。
        self.inputs_embeds: CpuGpuBuffer | None = None
        self.token_indices = self._make_buffer(self.max_num_tokens, torch.int64)
        self.req_indices = self._make_buffer(self.max_num_tokens, torch.int64)
        self.query_pos = self._make_buffer(self.max_num_tokens, torch.int64)
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, torch.int32)
        self.prev_positions = self._make_buffer(
            self.max_num_reqs, torch.int64, fill_value=-1
        )
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, torch.int32
        )
        self.prev_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, torch.int32
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, torch.int32, fill_value=-1
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, torch.int32, fill_value=1
        )

        # vLLM 直接在 GPU 上计算这三项，供 RoPE/Attention 使用。
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # 每个 KV group 一份；initialize_kv_cache 后才能确定 group 数和块大小。
        self.slot_mappings: dict[int, CpuGpuBuffer] = {}
        self.block_tables: dict[int, CpuGpuBuffer] = {}
        self._previous_batch_req_id_to_index: dict[str, int] = {}
        self.last_prepared_inputs: PreparedInputBuffers | None = None
        self.last_attention_metadata: FullAttentionMetadataCollection | None = None
        self.last_model_inputs: ModelForwardInputs | None = None

    def _make_buffer(
        self,
        shape,
        dtype: torch.dtype,
        *,
        fill_value: int | float | bool = 0,
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            shape, dtype=dtype, device=self.device, fill_value=fill_value
        )

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
        input_ids = torch.randint(0, vocab_size, (num_tokens,), device=self.device)
        positions = torch.arange(num_tokens, device=self.device, dtype=torch.int64)
        hidden_states = self.model(input_ids, positions)
        # vLLM 只对需要采样的位置算 logits。这里取最多 max_num_seqs 个末尾
        # hidden state，覆盖 logits/sampler 峰值但不引入 InputBatch 状态。
        num_sampled_positions = min(self.vllm_config.max_num_seqs, num_tokens)
        logits = self.model.compute_logits(hidden_states[-num_sampled_positions:])
        _ = torch.argmax(logits, dim=-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        del input_ids, positions, hidden_states, logits

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
        self.input_batch = InputBatch(
            max_num_reqs=self.vllm_config.max_num_seqs,
            max_model_len=self.vllm_config.max_model_len,
            num_kv_cache_groups=len(kv_cache_config.kv_cache_groups),
            pin_memory=self.device.type == "cuda",
        )
        self.slot_mappings = {
            group_id: self._make_buffer(self.max_num_tokens, torch.int64)
            for group_id in range(len(kv_cache_config.kv_cache_groups))
        }
        self.block_tables = {
            group_id: self._make_buffer(
                (
                    self.max_num_reqs,
                    (
                        self.vllm_config.max_model_len
                        + group.kv_cache_spec.block_size
                        - 1
                    )
                    // group.kv_cache_spec.block_size,
                ),
                torch.int32,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        logger.info(
            "KV Cache 物理初始化完成：layers=%d, blocks=%d, bytes=%.2f GiB",
            len(self.kv_caches),
            kv_cache_config.num_blocks,
            sum(t.size for t in kv_cache_config.kv_cache_tensors) / 1024**3,
        )

    def get_model(self) -> nn.Module | None:
        return self.model

    @torch.inference_mode()
    def model_forward(self, model_inputs: ModelForwardInputs) -> torch.Tensor:
        """直接执行已经 preprocess 好的扁平 Qwen2 输入。"""

        if self.model is None:
            raise RuntimeError("必须先 load_model，再执行真实模型前向")
        return self.model(
            input_ids=model_inputs.input_ids,
            positions=model_inputs.positions,
            attention_metadata=model_inputs.attention_metadata,
        )

    def _update_states(self, scheduler_output) -> None:
        """用 SchedulerOutput 增量更新请求缓存和持久 InputBatch。"""

        if self.input_batch is None:
            raise RuntimeError("必须先 initialize_kv_cache，再执行模型")

        # _prepare_inputs 用它表达“当前 batch 槽位 -> 上一 batch 槽位”。必须在
        # remove/condense/add 改变 InputBatch 之前拍快照。
        self._previous_batch_req_id_to_index = dict(
            self.input_batch.req_id_to_index
        )

        # 1. 已结束请求：完整状态与 batch 槽位一起释放。
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)

        # 2. 本轮未调度请求：只移出紧凑 batch，完整请求状态继续缓存。
        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
        unscheduled_req_ids = (
            set(self.input_batch.req_id_to_index) - scheduled_req_ids
        )
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)
        self.input_batch.condense()

        # 3. 新请求或抢占恢复请求携带完整快照，覆盖 Worker 的旧缓存。
        for new_req in scheduler_output.scheduled_new_reqs:
            req_state = CachedRequestState(
                req_id=new_req.req_id,
                prompt_token_ids=list(new_req.prompt_token_ids),
                output_token_ids=list(new_req.output_token_ids),
                sampling_params=new_req.sampling_params,
                block_ids=tuple(list(ids) for ids in new_req.block_ids),
                num_computed_tokens=new_req.num_computed_tokens,
            )
            self.requests[new_req.req_id] = req_state
            if new_req.req_id in self.input_batch.req_id_to_index:
                self.input_batch.remove_request(new_req.req_id)
                self.input_batch.condense()
            self.input_batch.add_request(req_state)

        # 4. 已缓存请求只接收 computed 进度和本轮新增 block id。
        cached = scheduler_output.scheduled_cached_reqs
        field_lengths = {
            len(cached.req_ids),
            len(cached.new_block_ids),
            len(cached.num_computed_tokens),
            len(cached.num_scheduled_tokens),
        }
        if len(field_lengths) != 1:
            raise ValueError("CachedRequestData 各字段长度不一致")

        for index, req_id in enumerate(cached.req_ids):
            try:
                req_state = self.requests[req_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Worker 没有请求 {req_id} 的缓存状态，无法应用增量"
                ) from exc

            req_state.num_computed_tokens = cached.num_computed_tokens[index]
            new_block_ids = cached.new_block_ids[index]
            if new_block_ids is not None:
                for block_ids, new_ids in zip(
                    req_state.block_ids, new_block_ids, strict=True
                ):
                    block_ids.extend(new_ids)

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                self.input_batch.add_request(req_state)
            else:
                self.input_batch.num_computed_tokens_cpu[req_index] = (
                    req_state.num_computed_tokens
                )
                if new_block_ids is not None:
                    self.input_batch.block_table.append_row(
                        new_block_ids, req_index
                    )

        if set(self.input_batch.req_ids) != scheduled_req_ids:
            raise RuntimeError(
                "InputBatch 与本轮调度请求不一致："
                f"batch={self.input_batch.req_ids}, "
                f"scheduled={list(scheduler_output.num_scheduled_tokens)}"
            )

    def _prepare_inputs(self, scheduler_output) -> PreparedInputBuffers:
        """把持久 InputBatch 整理成本轮扁平 CPU/GPU 工作缓冲。

        此方法止步于 token/位置/计数/slot mapping；Attention metadata 和
        ``model()`` 输入对象由后续阶段基于这些缓冲构建。
        """

        if self.input_batch is None or self.kv_cache_config is None:
            raise RuntimeError("必须先 initialize_kv_cache，再准备模型输入")

        req_ids = self.input_batch.req_ids
        num_reqs = len(req_ids)
        if num_reqs > self.max_num_reqs:
            raise RuntimeError("本轮请求数超过预分配容量")

        scheduled_counts = [
            int(scheduler_output.num_scheduled_tokens[req_id])
            for req_id in req_ids
        ]
        total_num_tokens = sum(scheduled_counts)
        if total_num_tokens != scheduler_output.total_num_scheduled_tokens:
            raise ValueError("SchedulerOutput total_num_scheduled_tokens 不一致")
        if total_num_tokens > self.max_num_tokens:
            raise RuntimeError("本轮 token 数超过预分配容量")

        # 请求级计数与前后 batch 槽位映射。
        cumulative = 0
        self.query_start_loc.np[0] = 0
        token_cursor = 0
        for req_index, (req_id, num_scheduled) in enumerate(
            zip(req_ids, scheduled_counts, strict=True)
        ):
            if num_scheduled <= 0:
                raise ValueError(f"请求 {req_id} 的 num_scheduled_tokens 必须大于 0")
            num_computed = int(
                self.input_batch.num_computed_tokens_cpu[req_index]
            )
            seq_len = num_computed + num_scheduled
            if seq_len > self.vllm_config.max_model_len:
                raise ValueError(f"请求 {req_id} 本轮 seq_len 超过 max_model_len")

            self.num_scheduled_tokens.np[req_index] = num_scheduled
            self.prev_positions.np[req_index] = (
                self._previous_batch_req_id_to_index.get(req_id, -1)
            )
            self.prev_num_draft_tokens.np[req_index] = 0
            self.num_decode_draft_tokens.np[req_index] = -1
            self.num_accepted_tokens.np[req_index] = int(
                self.input_batch.num_accepted_tokens_cpu[req_index]
            )

            cumulative += num_scheduled
            self.query_start_loc.np[req_index + 1] = cumulative
            for local_query_pos in range(num_scheduled):
                flat_index = token_cursor + local_query_pos
                absolute_position = num_computed + local_query_pos
                self.req_indices.np[flat_index] = req_index
                self.query_pos.np[flat_index] = local_query_pos
                self.token_indices.np[flat_index] = (
                    req_index * self.vllm_config.max_model_len
                    + absolute_position
                )
            token_cursor += num_scheduled

        # 固定容量缓冲的尾部也写成稳定哨兵值，方便 CUDA graph 后续复用。
        self.query_start_loc.np[num_reqs + 1 :].fill(total_num_tokens)
        self.prev_positions.np[num_reqs:].fill(-1)
        self.prev_num_draft_tokens.np[num_reqs:].fill(0)
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_scheduled_tokens.np[num_reqs:].fill(0)

        # 二维持久 token 表 -> 本轮一维 input_ids/is_token_ids。
        if total_num_tokens:
            token_indices_tensor = self.token_indices.cpu[:total_num_tokens]
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                token_indices_tensor,
                out=self.input_ids.cpu[:total_num_tokens],
            )
            torch.index_select(
                self.input_batch.is_token_ids_tensor.flatten(),
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_tokens],
            )

        # CPU 工作结果搬到执行设备。
        for buffer in (
            self.input_ids,
            self.is_token_ids,
            self.token_indices,
            self.req_indices,
            self.query_pos,
        ):
            buffer.copy_to_gpu(total_num_tokens)
        for buffer in (
            self.query_start_loc,
            self.prev_positions,
            self.num_scheduled_tokens,
            self.prev_num_draft_tokens,
            self.num_decode_draft_tokens,
            self.num_accepted_tokens,
        ):
            buffer.copy_to_gpu()

        if num_reqs:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=self.device.type == "cuda",
            )
            req_indices_gpu = self.req_indices.gpu[:total_num_tokens]
            self.positions[:total_num_tokens] = (
                self.num_computed_tokens[req_indices_gpu].to(torch.int64)
                + self.query_pos.gpu[:total_num_tokens]
            )
            self.seq_lens[:num_reqs] = (
                self.num_computed_tokens[:num_reqs]
                + self.num_scheduled_tokens.gpu[:num_reqs]
            )
        self.num_computed_tokens[num_reqs:].zero_()
        self.seq_lens[num_reqs:].zero_()

        # 写 KV 的地址：slot = physical_block_id * block_size + block_offset。
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            block_size = group.kv_cache_spec.block_size
            slot_buffer = self.slot_mappings[group_id]
            for flat_index in range(total_num_tokens):
                req_index = int(self.req_indices.np[flat_index])
                position = int(
                    self.input_batch.num_computed_tokens_cpu[req_index]
                    + self.query_pos.np[flat_index]
                )
                logical_block = position // block_size
                block_offset = position % block_size
                block_ids = self.input_batch.block_table[group_id][req_index]
                if logical_block >= len(block_ids):
                    raise RuntimeError(
                        "block table 无法覆盖本轮 token："
                        f"group={group_id}, req={req_ids[req_index]}, "
                        f"position={position}, blocks={block_ids}"
                    )
                slot_buffer.np[flat_index] = (
                    block_ids[logical_block] * block_size + block_offset
                )
            slot_buffer.copy_to_gpu(total_num_tokens)

        prepared = PreparedInputBuffers(
            num_reqs=num_reqs,
            num_tokens=total_num_tokens,
            input_ids=self.input_ids.gpu[:total_num_tokens],
            positions=self.positions[:total_num_tokens],
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            slot_mappings={
                group_id: buffer.gpu[:total_num_tokens]
                for group_id, buffer in self.slot_mappings.items()
            },
        )
        self.last_prepared_inputs = prepared
        return prepared

    def _build_attention_metadata(
        self, prepared: PreparedInputBuffers
    ) -> FullAttentionMetadataCollection:
        """把公共输入缓冲和各 group 地址表打包成 FullAttention metadata。"""

        if self.input_batch is None or self.kv_cache_config is None:
            raise RuntimeError("必须先 initialize_kv_cache，再构建 Attention metadata")
        if prepared.num_reqs <= 0 or prepared.num_tokens <= 0:
            raise ValueError("空 batch 不需要 Attention metadata")

        by_group: dict[int, FullAttentionMetadata] = {}
        by_layer: dict[str, FullAttentionMetadata] = {}
        query_lens_cpu = [
            int(value)
            for value in self.num_scheduled_tokens.np[: prepared.num_reqs]
        ]
        seq_lens_cpu = [
            int(self.input_batch.num_computed_tokens_cpu[req_index])
            + query_lens_cpu[req_index]
            for req_index in range(prepared.num_reqs)
        ]
        max_query_len = max(query_lens_cpu)
        max_seq_len = max(seq_lens_cpu)

        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                raise NotImplementedError(
                    "当前 metadata 只实现 FullAttention，"
                    f"收到 {type(spec).__name__}"
                )

            # InputBatch 保存 jagged Python block rows；Attention kernel 需要固定的
            # GPU 二维页表，因此在这里整理为 [num_reqs, max_num_blocks]。
            block_table_buffer = self.block_tables[group_id]
            block_table_buffer.np.fill(0)  # block 0 是保留的 null block。
            max_num_blocks = block_table_buffer.cpu.shape[1]
            for req_index in range(prepared.num_reqs):
                block_ids = self.input_batch.block_table[group_id][req_index]
                required_blocks = (
                    seq_lens_cpu[req_index] + spec.block_size - 1
                ) // spec.block_size
                if len(block_ids) < required_blocks:
                    raise RuntimeError(
                        "block table 无法覆盖 Attention 的 seq_len："
                        f"group={group_id}, req={self.input_batch.req_ids[req_index]}, "
                        f"required={required_blocks}, actual={len(block_ids)}"
                    )
                if len(block_ids) > max_num_blocks:
                    raise RuntimeError("请求 block table 超过预分配容量")
                if block_ids:
                    block_table_buffer.np[req_index, : len(block_ids)] = block_ids
            block_table_buffer.copy_to_gpu(prepared.num_reqs)

            metadata = FullAttentionMetadata(
                kv_cache_group_id=group_id,
                layer_names=tuple(group.layer_names),
                block_size=spec.block_size,
                causal=True,
                num_reqs=prepared.num_reqs,
                num_actual_tokens=prepared.num_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                query_start_loc=prepared.query_start_loc,
                seq_lens=prepared.seq_lens,
                num_computed_tokens=self.num_computed_tokens[
                    : prepared.num_reqs
                ],
                num_scheduled_tokens=self.num_scheduled_tokens.gpu[
                    : prepared.num_reqs
                ],
                positions=prepared.positions,
                block_table=block_table_buffer.gpu[: prepared.num_reqs],
                slot_mapping=prepared.slot_mappings[group_id],
            )
            by_group[group_id] = metadata
            for layer_name in group.layer_names:
                if layer_name in by_layer:
                    raise RuntimeError(f"Attention layer {layer_name} 属于多个 KV group")
                by_layer[layer_name] = metadata

        collection = FullAttentionMetadataCollection(
            by_group=by_group,
            by_layer=by_layer,
        )
        self.last_attention_metadata = collection
        return collection

    def _preprocess(
        self,
        prepared: PreparedInputBuffers,
        attention_metadata: FullAttentionMetadataCollection,
    ) -> ModelForwardInputs:
        """选择纯文本模型路径，产出扁平化的真实前向输入。"""

        if not bool(self.is_token_ids.gpu[: prepared.num_tokens].all().item()):
            raise NotImplementedError("当前 _preprocess 尚未接入 prompt embeddings")

        # 每个请求只需要最后一个 query hidden state 计算下一 token logits。
        logits_indices = prepared.query_start_loc[1:].to(torch.int64) - 1
        model_inputs = ModelForwardInputs(
            input_ids=prepared.input_ids,
            inputs_embeds=None,
            positions=prepared.positions,
            attention_metadata=attention_metadata,
            logits_indices=logits_indices,
        )
        self.last_model_inputs = model_inputs
        return model_inputs

    def _bookkeeping_after_sample(
        self,
        scheduler_output,
        req_ids: list[str],
        sampled_token_ids: list[list[int]],
    ) -> None:
        """模型执行成功后提交 computed 进度与新采样 token。"""

        assert self.input_batch is not None
        for req_id, sampled_ids in zip(
            req_ids, sampled_token_ids, strict=True
        ):
            req_state = self.requests[req_id]
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            req_state.num_computed_tokens += num_scheduled
            req_index = self.input_batch.req_id_to_index[req_id]
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_state.num_computed_tokens
            )
            if sampled_ids:
                req_state.output_token_ids.extend(sampled_ids)
                self.input_batch.append_output_token_ids(req_id, sampled_ids)

    def execute_model(self, scheduler_output):
        """执行一次同步的准备输入、模型前向和 greedy sampling。"""
        from my_vllm.v1.core.sched.output import ModelRunnerOutput

        self._update_states(scheduler_output)
        assert self.input_batch is not None
        model_inputs: ModelForwardInputs | None = None
        if scheduler_output.total_num_scheduled_tokens:
            prepared = self._prepare_inputs(scheduler_output)
            attention_metadata = self._build_attention_metadata(prepared)
            model_inputs = self._preprocess(prepared, attention_metadata)
        else:
            self.last_prepared_inputs = None
            self.last_attention_metadata = None
            self.last_model_inputs = None
        req_ids = self.input_batch.req_ids
        should_sample = [
            self.requests[req_id].num_computed_tokens
            + scheduler_output.num_scheduled_tokens[req_id]
            >= self.requests[req_id].num_tokens
            for req_id in req_ids
        ]

        if self.is_mock_model:
            # CPU 架构测试保留可读、确定的 token 序列。
            sampled_token_ids = [
                [ord("a") + len(self.requests[req_id].output_token_ids) % 26]
                if sample
                else []
                for req_id, sample in zip(req_ids, should_sample, strict=True)
            ]
        else:
            if model_inputs is None:
                sampled_token_ids = [[] for _ in req_ids]
            else:
                hidden_states = self.model_forward(model_inputs)
                sample_req_indices = [
                    index for index, sample in enumerate(should_sample) if sample
                ]
                sampled_token_ids = [[] for _ in req_ids]
                if sample_req_indices:
                    assert self.model is not None
                    selected_indices = model_inputs.logits_indices[
                        sample_req_indices
                    ]
                    logits = self.model.compute_logits(
                        hidden_states[selected_indices]
                    )
                    # baseline 只实现 greedy。argmax 后只 D2H 少量 token id，
                    # model forward 与 FlashAttention 内部没有 host 同步。
                    token_ids = torch.argmax(logits, dim=-1).to("cpu").tolist()
                    for req_index, token_id in zip(
                        sample_req_indices, token_ids, strict=True
                    ):
                        sampled_token_ids[req_index] = [int(token_id)]

        self._bookkeeping_after_sample(
            scheduler_output, req_ids, sampled_token_ids
        )
        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )
