"""
调度器 — Scheduler

对应 vLLM 的 vllm/v1/core/sched/scheduler.py（大幅简化版）。

核心思想（vLLM v1 的设计精髓）：
  没有「prefill 阶段」和「decode 阶段」之分。每个请求只有两个数字：
    - num_computed_tokens：已经过模型前向（KV 已算出）的 token 数
    - num_tokens：完整序列长度（prompt + 已生成输出），即「要算到哪」
  每步调度就是「让 computed 追赶 num_tokens」，在 token 预算内尽量多算。
  这个统一模型天然覆盖 chunked prefill、prefix cache、被抢占后恢复等场景。

每步调度（schedule）产出 SchedulerOutput；执行器前向后，update_from_output()
根据采样到的 token 推进请求状态、判定结束、释放 KV cache。

简化说明（相比 vLLM）：
  - 调度策略固定 FCFS（waiting 用 deque，先进先出），去掉优先级队列。
  - 抢占只选「本轮尚未调度」的 running 请求，避免复杂的索引回退。
  - 去掉投机解码、多模态 encoder、KV connector、chunked prefill 阈值等。
  - 结束判定简化为「达到 max_tokens / max_model_len」（真实 EOS 检测待 tokenizer 接入）。
"""

import logging
from collections import deque

from my_vllm.config import EngineConfig
from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    ModelRunnerOutput,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.request import Request, RequestStatus

logger = logging.getLogger(__name__)


class Scheduler:
    """调度器：每步选出哪些请求、分配多少 token 和 KV block

    Args:
        vllm_config: 引擎配置（携带 max_model_len / block_size / 并发上限等）。
        kv_cache_manager: KV cache 管理器（分配 / 释放 block）。
    """

    def __init__(self, vllm_config: EngineConfig, kv_cache_manager: KVCacheManager):
        self.vllm_config = vllm_config
        self.kv_cache_manager = kv_cache_manager

        self.max_model_len = vllm_config.max_model_len
        self.block_size = kv_cache_manager.scheduler_block_size
        self.max_num_running_reqs = vllm_config.max_num_seqs
        self.max_num_scheduled_tokens = vllm_config.max_num_batched_tokens

        # 全部请求：request_id -> Request（含已结束但尚未回收输出的）
        self.requests: dict[str, Request] = {}
        # 就绪队列（新请求 + 被抢占后待恢复的请求）
        self.waiting: deque[Request] = deque()
        # 正在执行的请求（已分配 KV cache，按加入顺序排列）
        self.running: list[Request] = []

        # 本轮/上轮之间结束的请求 id（由 EngineCore 消费后清空）
        self.finished_req_ids: set[str] = set()
        # Worker 也必须收到一次 finished 通知。EngineCore 会先消费上面的集合，
        # 因而不能让两个消费者共用同一个 set。
        self._finished_req_ids_to_notify: set[str] = set()

        self.current_step = 0

    # ==================================================================
    # 请求进出
    # ==================================================================

    def add_request(self, request: Request) -> None:
        """接收一个新请求，放入就绪队列"""
        # prompt 超长直接忽略（对应 vLLM 的 FINISHED_IGNORED）
        if request.num_prompt_tokens > self.max_model_len:
            request.status = RequestStatus.FINISHED_IGNORED
            self.requests[request.request_id] = request # 调度器的总表目记录一下
            self._mark_finished(request.request_id)
            logger.warning(
                "请求 %s prompt 超长（%d > %d），忽略",
                request.request_id, request.num_prompt_tokens, self.max_model_len,
            )
            return

        self.requests[request.request_id] = request # 调度器的总表目记录一下
        self.waiting.append(request)                # 加入waiting队列， 这里发过来的肯定都是没有处理过，都是prompt的req
        logger.debug("请求 %s 已入队 (num_prompt_tokens=%d)", request.request_id,
                     request.num_prompt_tokens)

    def finish_requests(self, request_ids: list[str]) -> None:
        """主动终止一批请求（如用户 cancel）"""
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            request.status = RequestStatus.FINISHED_ABORTED
            if request in self.waiting:
                self.waiting.remove(request)
            if request in self.running:
                self.running.remove(request)
            self.kv_cache_manager.free(request)
            self._mark_finished(req_id)

    # ==================================================================
    # 调度主流程
    # ==================================================================

    def schedule(self) -> SchedulerOutput:
        """执行一步调度，返回本轮要计算的内容

        阶段：
          1. RUNNING 遍历：延续已在跑的请求（decode 或 chunked prefill）。
          2. WAITING 遍历：从就绪队列拉新请求（含被抢占后恢复的）。
          3. 构造 SchedulerOutput。
        """
        self.current_step += 1

        scheduled_new_reqs: list[Request] = []      # 本轮的新req的名单
        scheduled_running_reqs: list[Request] = []  # 本轮继续的req的名单
        preempted_reqs: list[Request] = []          # 本轮被抢占掉的名单（不运行，回到waiting了）

        # 本轮新增的 block id，外层 tuple 的每一项对应一个 KV cache group。
        req_to_new_blocks: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}    # 本轮每个req所需要计算的token数（prompt长度的，chunked长度的，1token长度的）
        token_budget = self.max_num_scheduled_tokens # 本轮token预算

        # ---- 阶段 1：RUNNING 遍历 ----
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # 本轮要计算的新 token 数 = 要算到哪 - 已算到哪
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            if num_new_tokens <= 0:
                req_index += 1
                continue

            num_new_tokens = min(
                num_new_tokens,
                token_budget,
                self.max_model_len - request.num_computed_tokens,
            )
            if num_new_tokens <= 0:
                req_index += 1
                continue

            # 分配 KV cache；失败则抢占一个「本轮尚未调度」的 running 请求
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is not None:
                    break

                victim = self._pick_preempt_victim(scheduled_running_reqs) # 传入已经调度的表，是为了排除掉他们
                if victim is None:
                    new_blocks = None
                    break
                self.running.remove(victim)
                self._preempt_request(victim)
                preempted_reqs.append(victim)
                if victim is request:
                    # 抢占到自己头上，说明无他人可抢，本轮放弃该请求
                    new_blocks = None
                    break

            if new_blocks is None:
                break  # 无法再调度 running 请求

            scheduled_running_reqs.append(request)
            req_id = request.request_id
            req_to_new_blocks[req_id] = new_blocks.get_block_ids()
            num_scheduled_tokens[req_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        # ---- 阶段 2：WAITING 遍历（本轮有抢占则跳过，避免重复调度）----
        if not preempted_reqs:
            while (
                self.waiting # 全新的req, 被抢占过来的req:(prompt+output, chunked_prompt, prompt)
                and token_budget > 0
                and len(self.running) < self.max_num_running_reqs
            ):
                request = self.waiting[0]
                req_id = request.request_id

                '''
                waiting队列里面只有3种req
                请求类型	              KV cache	   num_computed_tokens
                从未调度过	                无	            == 0
                被抢占赶回（recompute）	    已释放	         == 0（2038 行清零）
                PD 分离·远端 KV 传输完成	有（远端搬入）	  != 0

            
                你原来的两分法漏了第三类。num_computed_tokens != 0 说的不是「有 prompt+output 但没 KV」的抢占请求，
                而是「KV 已经通过 connector 从 prefill 实例搬过来了」的 disaggregation 请求——它的 KV 是在的。

                你正在看的 partial_tail_offloads（output.py:312）也属于这条 KV connector 链路。
                '''
                # 前缀缓存查询（仅第一次调度；被抢占恢复的请求 computed 已归零）
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_hit_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                else: # ----- 这里是给KVTransfer PD分离用的 -----
                    # 这里num_computed_tokens !=0, 表示这个req的prefill的kvcache,在远端计算好了
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_hit_tokens = 0

                num_computed_tokens = num_hit_tokens # prefix前缀缓存命中的token数
                num_new_tokens = request.num_tokens - num_computed_tokens # 后续还要计算的token数
                num_new_tokens = min(
                    num_new_tokens,
                    token_budget,
                    self.max_model_len - num_computed_tokens,
                )
                if num_new_tokens <= 0:
                    break  # 预算耗尽或超出上下文，本轮不再拉新

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens, # 需要计算的tokens
                    num_new_computed_tokens=num_hit_tokens,
                    new_computed_blocks=new_computed_blocks,
                )
                if new_blocks is None:
                    break  # 显存不足，本轮不再拉新

                # 出队 → 进入 running
                self.waiting.popleft()
                self.running.append(request)
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens  # 前缀命中数

                scheduled_new_reqs.append(request)
                # 新请求要发完整 block 表（含前缀命中的 block）
                req_to_new_blocks[req_id] = self.kv_cache_manager.get_block_ids(
                    req_id
                )
                num_scheduled_tokens[req_id] = num_new_tokens # 本轮的新req，需要完成prefix除了命中的剩余的tokens数
                token_budget -= num_new_tokens

        # ---- 阶段 3：构造输出 ----
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        new_reqs_data = [ # 本轮新增的req的请求集合
            NewRequestData(
                req_id=r.request_id, # 每个请求的ID
                prompt_token_ids=r.prompt_token_ids, # 每个req的 原始 prompt 的 token ids 序列 #
                output_token_ids=list(r.output_token_ids),
                sampling_params=r.sampling_params,
                block_ids=req_to_new_blocks[r.request_id], # 每个group的完整block table
                num_computed_tokens=r.num_computed_tokens, # 每个req的已经计算的tokens数量
                max_tokens=r.max_tokens, # 这个req的最大总长度
            )
            for r in scheduled_new_reqs
        ]

        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs, num_scheduled_tokens, req_to_new_blocks
            # 本轮继续运行的req集合，每个req的本轮的计算tokens数量，每个req的本轮新增的block列表
        )

        finished_req_ids = set(self._finished_req_ids_to_notify)
        self._finished_req_ids_to_notify.clear()
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data, # 本轮新增的req的集合
            scheduled_cached_reqs=cached_reqs_data, # 本轮的继续running的req集合
            num_scheduled_tokens=num_scheduled_tokens, # 这边传入的本轮的每个req需要计算的token数

            total_num_scheduled_tokens=total_num_scheduled_tokens, # batch的总共tokens数
            finished_req_ids=finished_req_ids,

            # 这一批的req之间，互相共享的 公共前缀长度；不是各自命中前缀缓存的最大值
            # 这里是为了cascade attention， 这是本轮batch，他们都面临同样的K矩阵，V矩阵，对于这个公共前缀长度

            # num_common_prefix_blocks 这个量本身是 prefix cache 的产物（共享 block 的计数），但它服务的用途是 cascade attention（省 attention 算力）。
            # 两者解决的是 attention 流水线上不同阶段的两段重复——一个是「KV 别重复生成」，一个是「共享的 KV 别重复 attend」。
            num_common_prefix_blocks=self.kv_cache_manager.get_num_common_prefix_blocks( # running 队列里所有请求共同的最长公共前缀 block 数
                self.running[0].request_id if self.running else ""
            ),
        )

        # 调度后：推进每个请求的 num_computed_tokens
        self._update_after_schedule(scheduler_output)

        return scheduler_output

    # ==================================================================
    # 输出消费
    # ==================================================================

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[str, list[int]]:
        """根据执行器采样结果推进请求，返回 {request_id: 本轮新生成 token}

        对每个被调度的请求：追加采到的 token，判定是否结束，结束则释放 KV cache。
        """
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index

        outputs: dict[str, list[int]] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            idx = req_id_to_index[req_id]
            new_token_ids = sampled[idx] if sampled else []
            outputs[req_id] = new_token_ids

            # schedule 阶段只是预留 block；执行器返回后 KV 才真实写好。
            # 因此 Prefix Cache 必须在这里登记，并且只能登记已定稿的 full block。
            num_tokens_to_cache = min(
                request.num_computed_tokens,
                request.num_tokens,
            )
            self.kv_cache_manager.cache_blocks(request, num_tokens_to_cache)

            stopped = self._update_request_with_output(request, new_token_ids)
            if stopped:
                self.kv_cache_manager.free(request)
                self._mark_finished(req_id)
                if request in self.running:
                    self.running.remove(request)

        return outputs

    def _mark_finished(self, req_id: str) -> None:
        """分别登记给 EngineCore 输出消费者和 Worker 状态消费者。"""

        self.finished_req_ids.add(req_id)
        self._finished_req_ids_to_notify.add(req_id)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> bool:
        """追加生成 token 并判定是否结束，返回 True 表示本请求结束"""
        request.append_output_token_ids(new_token_ids)

        # 简化：无 EOS 检测（真实 tokenizer 接入后补充），只按长度停止
        if request.num_output_tokens >= request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        if request.num_tokens >= self.max_model_len:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        return False

    # ==================================================================
    # 内部辅助
    # ==================================================================
    # 先暂时更新
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """调度已提交，推进每个请求的 num_computed_tokens

        放在构造 SchedulerOutput 之后：output 里记录的是「本轮调度时的原始值」，
        这里再 +num_scheduled，这样下一轮 schedule 看到的是最新 computed。
        """
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled # 算上这次已经计算的

    def _pick_preempt_victim(
        self, scheduled_running_reqs: list[Request]
    ) -> Request | None:
        """从 running 队列尾部找一个「本轮尚未调度」的请求作为抢占 victim

        只抢未调度的请求：它们占用 block 但不在本轮输出里，抢它们无需回退索引。
        从尾部找 = 抢占「最新」的请求（FCFS 抢占语义）。
        """
        scheduled_ids = {id(r) for r in scheduled_running_reqs}
        for request in reversed(self.running):
            if id(request) not in scheduled_ids:
                return request
        return None

    def _preempt_request(self, request: Request) -> None:
        """抢占一个请求：释放 KV cache，状态置 PREEMPTED，放回就绪队列

        简化：computed 归零意味着恢复时从头重算（vLLM 会做增量重算，此处从简）。
        """
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        self.waiting.appendleft(request)  # 队头优先恢复

    def _make_cached_request_data(
        self,
        scheduled_running_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        req_to_new_blocks: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        """构造老请求的增量数据"""
        req_ids = [r.request_id for r in scheduled_running_reqs] # 本轮继续的req的集合
        new_block_ids = [req_to_new_blocks[r.request_id] for r in scheduled_running_reqs] # 各group本轮新增block
        num_computed = [r.num_computed_tokens for r in scheduled_running_reqs] # 本轮继续的req的已经计算过kvcache的token数
        num_scheduled = [num_scheduled_tokens[r.request_id] for r in scheduled_running_reqs] # 本轮继续req，在这一轮需要计算的token数
        return CachedRequestData(
            req_ids=req_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed,
            num_scheduled_tokens=num_scheduled,
        )
