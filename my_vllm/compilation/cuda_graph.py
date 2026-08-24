"""CUDA Graph 的描述符、调度器、图条目与模型 wrapper。

本模块按 vLLM 的职责边界拆分：

* :class:`CUDAGraphDispatcher` 只管理合法 ``(mode, BatchDescriptor)`` 库；
* :class:`CUDAGraphWrapper` 只管理真实图条目并执行 capture/replay；
* :class:`CUDAGraphEntry` 保存一张图、静态输出和输入地址。

第一阶段仍只实现纯 Decode 的 FULL CUDA Graph。Prefill、混合 batch 以及未命中
合法描述符的 Decode batch 都明确回退 eager。
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from my_vllm.worker.gpu_model_runner import ModelForwardInputs

logger = logging.getLogger(__name__)


class CUDAGraphMode(str, Enum):
    """一次模型调用采用的 CUDA Graph 运行模式。"""

    NONE = "NONE"
    FULL = "FULL"


@dataclass(frozen=True, order=True)
class BatchDescriptor:
    """一张静态执行图对应的 batch 规格。

    ``is_uniform`` 表示每个请求本轮 query 长度相同。当前 FULL 模式进一步要求
    它是纯 Decode，因此 ``num_tokens == num_reqs`` 且每个请求只有一个 query。
    ``max_seq_len`` 是 Attention kernel 捕获时固定的 K/V 循环上界。
    """

    num_tokens: int
    num_reqs: int
    max_seq_len: int
    is_uniform: bool


@dataclass
class CUDAGraphEntry:
    """Wrapper 持有的一张已经捕获的 CUDA Graph。"""

    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    input_addresses: tuple[int, ...]


class CUDAGraphDispatcher:
    """管理合法 ``(mode, BatchDescriptor)``，不持有真实 CUDA Graph。

    Dispatcher 在 ``GPUModelRunner.__init__`` 时还是空的；等 KV Cache 和
    Attention backend 的静态规格确定后，才由 ``initialize_cudagraph_keys``
    建立合法 key 库。这一点与图捕获本身严格分离。
    """

    def __init__(
        self,
        *,
        enabled: bool,
        max_num_reqs: int,
        max_model_len: int,
        capture_batch_sizes: Iterable[int],
        seq_len_bucket_size: int,
    ) -> None:
        if seq_len_bucket_size <= 0:
            raise ValueError("seq_len_bucket_size 必须大于 0")
        self.enabled = enabled
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.capture_batch_sizes = tuple(
            sorted(
                {
                    int(size)
                    for size in capture_batch_sizes
                    if 0 < int(size) <= max_num_reqs
                }
            )
        )
        self.seq_len_bucket_size = seq_len_bucket_size
        self.valid_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.FULL: set(),
        }
        self._seq_len_capture_sizes: tuple[int, ...] = ()
        self.is_initialized = False

    def initialize_cudagraph_keys(self) -> None:
        """在 KV Cache 初始化后建立纯 Decode FULL 的合法描述符库。"""

        self.valid_keys[CUDAGraphMode.FULL].clear()
        if not self.enabled:
            self.is_initialized = True
            return
        if not self.capture_batch_sizes:
            raise ValueError("启用 CUDA Graph 时至少需要一个 capture batch 档位")

        # Attention kernel 把 max_seq_len 固化在 graph launch 参数中。使用固定
        # 步长档位，避免 1025 token 被映射到 2048 后产生接近一倍的无效 KV
        # 扫描。max_model_len 不是步长整数倍时仍作为最后一个合法档位加入。
        sizes = list(
            range(
                min(self.seq_len_bucket_size, self.max_model_len),
                self.max_model_len + 1,
                self.seq_len_bucket_size,
            )
        )
        sizes.append(self.max_model_len)
        self._seq_len_capture_sizes = tuple(sorted(set(sizes)))

        for num_reqs in self.capture_batch_sizes:
            for max_seq_len in self._seq_len_capture_sizes:
                self.valid_keys[CUDAGraphMode.FULL].add(
                    BatchDescriptor(
                        num_tokens=num_reqs,
                        num_reqs=num_reqs,
                        max_seq_len=max_seq_len,
                        is_uniform=True,
                    )
                )
        self.is_initialized = True
        logger.info(
            "【CUDA Graph Dispatcher】合法描述符库初始化完成："
            "mode=FULL, batch_sizes=%s, seq_len_sizes=%s, keys=%d",
            self.capture_batch_sizes,
            self._seq_len_capture_sizes,
            len(self.valid_keys[CUDAGraphMode.FULL]),
        )

    def dispatch(
        self,
        *,
        num_tokens: int,
        num_reqs: int,
        max_seq_len: int,
        is_uniform_decode: bool,
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None]:
        """为真实 batch 选择 FULL key；无法合法映射时返回 eager。"""

        if (
            not self.enabled
            or not self.is_initialized
            or not is_uniform_decode
            or num_tokens != num_reqs
            or num_reqs not in self.capture_batch_sizes
        ):
            return CUDAGraphMode.NONE, None
        index = bisect_left(self._seq_len_capture_sizes, max_seq_len)
        if index == len(self._seq_len_capture_sizes):
            return CUDAGraphMode.NONE, None
        descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_seq_len=self._seq_len_capture_sizes[index],
            is_uniform=True,
        )
        if descriptor not in self.valid_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.NONE, None
        return CUDAGraphMode.FULL, descriptor

    def get_capture_descriptors(
        self,
    ) -> list[tuple[CUDAGraphMode, BatchDescriptor]]:
        """返回启动阶段要主动捕获的全部描述符，较大的图优先。"""

        if not self.is_initialized:
            raise RuntimeError("必须先 initialize_cudagraph_keys")
        return sorted(
            (
                (mode, descriptor)
                for mode, descriptors in self.valid_keys.items()
                for descriptor in descriptors
            ),
            key=lambda item: (
                item[1].num_tokens,
                item[1].max_seq_len,
            ),
            reverse=True,
        )


class CUDAGraphWrapper:
    """包装模型 forward，按 Dispatcher 给出的描述符 capture 或 replay。"""

    def __init__(
        self,
        runnable: Callable[["ModelForwardInputs"], torch.Tensor],
        *,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDAGraphWrapper 只支持 CUDA device")
        self.runnable = runnable
        self.device = device
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.entries: dict[
            tuple[CUDAGraphMode, BatchDescriptor], CUDAGraphEntry
        ] = {}
        self.capture_count = 0
        self.replay_count = 0
        # 启动 capture_model 完成后关闭，防止真实请求偷偷创建新图。
        self.capturing_enabled = True

    @staticmethod
    def _input_addresses(
        model_inputs: "ModelForwardInputs",
    ) -> tuple[int, ...]:
        addresses = [
            model_inputs.input_ids.data_ptr(),
            model_inputs.positions.data_ptr(),
        ]
        for group_id in sorted(model_inputs.attention_metadata.by_group):
            metadata = model_inputs.attention_metadata.by_group[group_id]
            addresses.extend(
                tensor.data_ptr()
                for tensor in (
                    metadata.query_start_loc,
                    metadata.seq_lens,
                    metadata.num_computed_tokens,
                    metadata.num_scheduled_tokens,
                    metadata.positions,
                    metadata.block_table,
                    metadata.slot_mapping,
                )
            )
        return tuple(addresses)

    @torch.inference_mode()
    def __call__(
        self,
        model_inputs: "ModelForwardInputs",
        *,
        mode: CUDAGraphMode,
        descriptor: BatchDescriptor | None,
        is_graph_capturing: bool = False,
    ) -> torch.Tensor:
        if mode is CUDAGraphMode.NONE or descriptor is None:
            return self.runnable(model_inputs)

        key = (mode, descriptor)
        addresses = self._input_addresses(model_inputs)
        entry = self.entries.get(key)
        if entry is None:
            if not is_graph_capturing or not self.capturing_enabled:
                logger.warning("CUDA Graph 未捕获，回退 eager：mode=%s, key=%s", mode, descriptor)
                return self.runnable(model_inputs)
            entry = self._capture(model_inputs, addresses)
            self.entries[key] = entry
            self.capture_count += 1
            logger.info("【CUDA Graph Wrapper】主动捕获完成：mode=%s, key=%s", mode.value, descriptor)
            return entry.output

        if addresses != entry.input_addresses:
            raise RuntimeError(
                "CUDA Graph replay 输入地址发生变化："
                f"key={key}, expected={entry.input_addresses}, actual={addresses}"
            )
        entry.graph.replay()
        self.replay_count += 1
        return entry.output

    def _capture(
        self,
        model_inputs: "ModelForwardInputs",
        input_addresses: tuple[int, ...],
    ) -> CUDAGraphEntry:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=capture_stream):
            output = self.runnable(model_inputs)
        current_stream.wait_stream(capture_stream)
        # capture 是否产生可消费输出在不同 PyTorch 版本中不应成为接口假设。
        graph.replay()
        return CUDAGraphEntry(
            graph=graph,
            output=output,
            input_addresses=input_addresses,
        )

    def finish_capture(self) -> None:
        """启动主动捕获结束后禁止运行期 capture-on-miss。"""

        self.capturing_enabled = False

    def clear(self) -> None:
        self.entries.clear()
