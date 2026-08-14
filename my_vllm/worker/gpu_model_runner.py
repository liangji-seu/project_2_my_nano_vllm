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

        # 每个请求已生成的输出 token 数（mock 采样用，模拟 worker 侧逐请求的生成状态）
        self._req_output_counts: dict[str, int] = {}

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
        """执行一次前向 + 采样（当前为 mock 占位，供 collective_rpc 链路跑通）

        对应 vLLM 的 GPUModelRunner.execute_model()。真实实现是把 scheduler_output
        的输入喂给模型 forward + 采样；本方法运行在 worker 进程内，由 EngineCore
        通过 executor.collective_rpc 广播调用。

        骨架阶段不跑真实模型，用确定性 mock 采样占位：每个请求每步生成一个可打印
        字符（按该请求已生成输出数轮转 a..z），保证主循环能跑通、请求能正常结束。
        用 self._req_output_counts 维护 worker 侧逐请求的「已生成输出数」，模拟
        真实 worker 为每个请求保存的生成状态。
        """
        from my_vllm.v1.core.sched.output import ModelRunnerOutput

        # 本轮被调度的请求 id（新/老请求统一从 num_scheduled_tokens 取）
        req_ids = list(scheduler_output.num_scheduled_tokens.keys())

        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            # 确定性「采样」：按已生成输出数轮转 a..z，避免 control 字符污染输出
            count = self._req_output_counts.get(req_id, 0)
            token_id = ord("a") + (count % 26)
            sampled_token_ids.append([token_id])
            self._req_output_counts[req_id] = count + 1

        # 释放已结束请求的本地计数（对应 worker 释放缓存状态的语义）
        for req_id in scheduler_output.finished_req_ids:
            self._req_output_counts.pop(req_id, None)

        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )
