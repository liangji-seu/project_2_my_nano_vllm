"""ModelRunner V2 风格的 PP sampled-token 旁路通信。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist

from my_vllm.distributed.parallel_state import (
    get_pipeline_model_parallel_world_size,
    get_pp_broadcast_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_last_stage,
)


@dataclass
class PendingRecv:
    event: torch.cuda.Event
    sampled_tokens: torch.Tensor
    req_ids: tuple[str, ...]
    should_sample: tuple[bool, ...]
    output_lengths: tuple[int, ...]


class PPTokenHandler:
    """在独立 stream/communicator 上收发 token，并延迟 ``pp_size`` 步消费。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.pp_size = get_pipeline_model_parallel_world_size()
        self.is_last = is_pipeline_last_stage()
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device=device)
        self.broadcast_group = get_pp_broadcast_group()
        self.last_rank = (
            (self.pp_size - 1)
            * get_tensor_model_parallel_world_size()
            + get_tensor_model_parallel_rank()
        )
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last else deque([None] * self.pp_size)
        )

    def begin_step(self) -> PendingRecv | None:
        """弹出 ``pp_size`` 步前的接收，并为当前 step 预留 FIFO 槽位。"""

        if self.is_last:
            return None
        pending = self.queue.popleft()
        self.queue.append(None)
        if pending is not None:
            self.main_stream.wait_event(pending.event)
        return pending

    def receive(
        self,
        req_ids: list[str],
        should_sample: list[bool],
        output_lengths: list[int],
    ) -> None:
        if self.is_last or not any(should_sample):
            return
        assert self.broadcast_group is not None
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                len(req_ids), dtype=torch.int64, device=self.device
            )
            dist.broadcast(
                sampled_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            event = self.broadcast_stream.record_event()
            sampled_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event=event,
            sampled_tokens=sampled_tokens,
            req_ids=tuple(req_ids),
            should_sample=tuple(should_sample),
            output_lengths=tuple(output_lengths),
        )

    def broadcast(self, sampled_tokens: torch.Tensor, should_sample: list[bool]) -> None:
        if not self.is_last or not any(should_sample):
            return
        assert self.broadcast_group is not None
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            dist.broadcast(
                sampled_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            sampled_tokens.record_stream(self.broadcast_stream)
