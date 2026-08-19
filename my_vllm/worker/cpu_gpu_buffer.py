"""固定地址的 CPU/GPU 双份工作缓冲区。"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CpuGpuBuffer:
    """CPU 负责填充，GPU 负责消费的预分配缓冲区。

    CUDA 环境下 CPU tensor 使用 pinned memory，使 ``copy_(non_blocking=True)``
    可以异步执行；CPU 调试环境仍保留两份 tensor，便于测试数据搬运边界。
    """

    def __init__(
        self,
        shape: int | Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        fill_value: int | float | bool = 0,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.device = device
        self.cpu = torch.full(
            tuple(shape),
            fill_value,
            dtype=dtype,
            device="cpu",
            pin_memory=device.type == "cuda",
        )
        self.gpu = torch.full(
            tuple(shape), fill_value, dtype=dtype, device=device
        )
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, count: int | None = None) -> None:
        """把完整缓冲区或第一维前 ``count`` 项复制到执行设备。"""

        if count is None:
            self.gpu.copy_(self.cpu, non_blocking=self.device.type == "cuda")
            return
        if count < 0 or count > self.cpu.shape[0]:
            raise ValueError(
                f"copy count 越界：count={count}, capacity={self.cpu.shape[0]}"
            )
        self.gpu[:count].copy_(
            self.cpu[:count], non_blocking=self.device.type == "cuda"
        )
