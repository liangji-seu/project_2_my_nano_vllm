"""
请求对象 — Request + RequestStatus + SamplingParams

对应 vLLM 的 vllm/v1/request.py（大幅简化版）。

一个 Request 就是一个待生成/生成中的推理请求，Scheduler 每步都在推进它的
`num_computed_tokens`，直到其状态变为 FINISHED_*。

简化说明（相比 vLLM）：
  - 去掉多模态 mm_features、结构化输出、LoRA、投机解码、pooling 等字段。
  - 去掉 torch 依赖（不保留 prompt_embeds），纯 Python 可运行、可测试。
"""

import enum
import hashlib
import struct
import time
from dataclasses import dataclass

from my_vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class SamplingParams:
    """采样参数（简化版）

    对应 vLLM 的 SamplingParams。只保留调度器需要的 max_tokens，
    其余采样细节（temperature / top_p / top_k 等）预留给 ModelRunner 采样阶段。
    """

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class FinishReason(enum.Enum):
    """请求结束原因（对应 vLLM 的 FinishReason）"""

    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"


class RequestStatus(enum.IntEnum):
    """请求状态（对应 vLLM 的 RequestStatus）

    约定：值 > PREEMPTED 的都算「已结束」状态，这是 is_finished() 的判据。
    """

    WAITING = enum.auto()          # 等待调度（新请求默认状态）
    RUNNING = enum.auto()          # 正在执行，持有 KV cache
    PREEMPTED = enum.auto()        # 被抢占，需后续恢复
    # ---- 以下均为结束状态 ----
    FINISHED_STOPPED = enum.auto()        # 正常停止（EOS）
    FINISHED_LENGTH_CAPPED = enum.auto()  # 达到 max_tokens / max_model_len
    FINISHED_ABORTED = enum.auto()        # 主动取消
    FINISHED_IGNORED = enum.auto()        # 忽略（如 prompt 超长）
    FINISHED_ERROR = enum.auto()          # 推理出错

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
}


class Request:
    """推理请求（简化版）

    核心状态：
      - num_computed_tokens：已经过模型前向（KV 已算出）的 token 数。
      - num_tokens：all_token_ids 长度 = prompt + 已生成输出，即「要算到哪」。
      每步调度就是让 computed 追赶 num_tokens，两者相等且命中停止条件即结束。
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        arrival_time: float | None = None,
        priority: int = 0,
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.arrival_time = arrival_time if arrival_time is not None else time.time()
        self.priority = priority

        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0
        self.num_preemptions = 0

        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.max_tokens = self.sampling_params.max_tokens

        # 内部列表：输出 token 与「完整 token 序列」（prompt + 输出）
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()

        # Request 维护「内容 -> 链式 hash」，KVCacheManager 维护
        # 「链式 hash + group id -> block」。Request 不保存任何物理 block id。
        self.block_hashes: list[BlockHash] = []
        self._block_hash_size: int | None = None

    # ---- 只读视图（禁止外部直接 append，必须走 append_output_token_ids）----

    @property
    def output_token_ids(self) -> list[int]:
        return self._output_token_ids

    @property
    def all_token_ids(self) -> list[int]:
        return self._all_token_ids

    @property
    def num_tokens(self) -> int:
        """完整序列长度 = prompt + 已生成输出"""
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    # ---- 追加输出 token ----

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        """追加一个或多个生成 token，同时更新 all_token_ids"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
        if self._block_hash_size is not None:
            self.update_block_hashes(self._block_hash_size)

    def update_block_hashes(self, block_size: int) -> None:
        """为所有新增 full block 计算确定性的链式内容 hash。

        第 N 块的 hash 同时包含第 N-1 块 hash 和本块 token，因此相同的局部
        token 片段只有在前面的完整 prefix 也相同时才会命中。
        """

        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        if self._block_hash_size not in (None, block_size):
            raise ValueError("同一个 Request 不能切换 block hash 粒度")
        self._block_hash_size = block_size

        num_full_blocks = self.num_tokens // block_size
        previous_hash = (
            bytes(self.block_hashes[-1]) if self.block_hashes else bytes(32)
        )
        for block_index in range(len(self.block_hashes), num_full_blocks):
            start = block_index * block_size
            token_ids = self._all_token_ids[start : start + block_size]
            token_bytes = b"".join(struct.pack(">q", token) for token in token_ids)
            previous_hash = hashlib.sha256(previous_hash + token_bytes).digest()
            self.block_hashes.append(BlockHash(previous_hash))

    # ---- 结束判定 ----

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def __lt__(self, other: "Request") -> bool:
        """按 priority → arrival_time → request_id 排序（用于优先级队列）"""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, status={self.status.name}, "
            f"num_computed={self.num_computed_tokens}, num_tokens={self.num_tokens})"
        )
