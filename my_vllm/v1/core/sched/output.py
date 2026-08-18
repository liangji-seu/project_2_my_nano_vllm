"""
调度输出数据结构 — SchedulerOutput + NewRequestData + CachedRequestData + ModelRunnerOutput

对应 vLLM 的 vllm/v1/core/sched/output.py（大幅简化版）。

Scheduler 每步产出 SchedulerOutput，交给执行器；执行器前向 + 采样后返回
ModelRunnerOutput，再由 Scheduler.update_from_output() 消费。

简化说明（相比 vLLM）：
  - NewRequestData 保留 sampling_params 和完整 token 快照；去掉 LoRA / 多模态等字段。
  - CachedRequestData 去掉投机解码、PP 的 new_token_ids 等。
  - SchedulerOutput 去掉 encoder / spec decode / KV connector 等字段。
  - ModelRunnerOutput 从 vllm/v1/outputs.py 挪到这里，只保留采样 token。
"""

from dataclasses import dataclass, field

from my_vllm.v1.request import FinishReason, SamplingParams


@dataclass
class NewRequestData:
    """首次调度或抢占恢复的请求（向 Worker 发送完整快照）。"""

    req_id: str
    prompt_token_ids: list[int]
    # 抢占恢复时不能只重发 prompt；Worker 重建持久状态还需要此前已生成的 token。
    output_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]  # 完整 block 表（含前缀命中的 block）
    num_computed_tokens: int          # 已计算 token 数（前缀命中数）
    max_tokens: int


@dataclass
class CachedRequestData:
    """本轮继续执行的老请求（worker 已有其元数据，只发增量）"""

    req_ids: list[str]
    new_block_ids: list[tuple[list[int], ...] | None]  # 本轮新增 block id（None 表示无新增）
    num_computed_tokens: list[int]
    num_scheduled_tokens: list[int]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls([], [], [], [])


@dataclass
class SchedulerOutput:
    """调度器一步的输出

    对应 vLLM 的 SchedulerOutput（简化版）。
    """

    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    # 每个请求本轮调度（要计算）的 token 数
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    # 上一步到这一步之间已结束的请求 id（供 worker 释放缓存状态）
    finished_req_ids: set[str]
    # 公共前缀 block 数（cascade attention 用，简化版恒为 0）
    num_common_prefix_blocks: list[int] = field(default_factory=list)

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
        )


@dataclass
class ModelRunnerOutput:
    """模型执行器一次前向的输出

    对应 vLLM 的 vllm/v1/outputs.py 里的 ModelRunnerOutput（简化版）。
    只保留「每个请求采到的 token」，logprobs / pooler 输出等一律省略。
    """

    req_ids: list[str]
    # sampled_token_ids[i] = req_ids[i] 本轮采出的新 token 列表（非投机解码时长度为 1）
    sampled_token_ids: list[list[int]]
    req_id_to_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.req_id_to_index:
            self.req_id_to_index = {rid: i for i, rid in enumerate(self.req_ids)}


@dataclass
class EngineCoreOutput:
    """一个请求结束时的输出（token id 级别，detokenize 由前端 renderer 负责）"""

    request_id: str
    token_ids: list[int]
    finish_reason: FinishReason | None
