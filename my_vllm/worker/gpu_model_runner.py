"""
GPUModelRunner 骨架 — Worker 内部的模型执行器

对应 vLLM 的 vllm/v1/worker/gpu_model_runner.py

当前阶段：只搭骨架，让 Worker 初始化链路能跑通，不实现真实推理。

后续 commit 实现（这是 vLLM 里最核心、最复杂的一个类）：
  - 模型加载：按模型名/路径构造模型结构，按 TP/PP 切分权重并放置到对应 GPU
  - 输入批处理：构造 InputBatch（把 scheduler 输出的 token 喂给模型）
  - forward 执行：模型前向 + 采样（sample）
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GPUModelRunner:
    """模型执行器（骨架）

    职责（预留）：
      - 持有模型实例 self.model
      - 加载权重并放到 GPU
      - 把 scheduler 输出的输入 batch 喂给模型，执行 forward
    """

    def __init__(self, vllm_config, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config  # 简化：直接用 EngineConfig 充当 model_config
        self.device = device

        # 模型实例，load_model() 后填充
        self.model: nn.Module | None = None

        logger.info("GPUModelRunner 骨架已构造 (device=%s)", device)

    def load_model(self) -> None:
        """加载模型权重（TODO：暂未实现）

        对应 vLLM 的 model_loader.load_model()：
          - 根据模型名/路径构造模型结构
          - 按 TP/PP 切分权重并放置到 self.device
          - model.eval()
        """
        # TODO: 真实加载模型权重。骨架阶段先置空，让初始化链路能跑通。
        self.model = None
        logger.warning(
            "GPUModelRunner.load_model() 未实现（骨架），self.model=None"
        )

    def get_model(self) -> nn.Module | None:
        """返回模型实例"""
        return self.model

    def execute_model(self, scheduler_output):
        """执行一次前向（TODO：暂未实现）"""
        raise NotImplementedError(
            "GPUModelRunner.execute_model() 未实现（骨架）"
        )
