# 8 月 16 日 ModelRunner 阶段面试题

> 范围：模型实例化、Hugging Face 权重文件、safetensors 权重加载、profiling run、KV Cache 物理显存初始化。
>
> 暂不涉及：`SchedulerOutput -> InputBatch` 的持久化批状态、真实推理 forward、采样。

## 一、先画出完整链路

1. 请不看代码，画出从 `EngineCore` 创建执行器开始，到 Scheduler 拿到 `KVCacheConfig` 为止的调用链。为什么 Scheduler 不能先于 Worker profiling 完成初始化？
2. `GPUWorker` 和 `GPUModelRunner` 的职责边界是什么？模型实例、模型权重占用、profiling、KV Cache 张量分别应该由谁持有？
3. 为什么模型加载发生在 Worker 进程，而不能由 EngineCore 加载一次后直接传给多个 Worker？
4. 在 TP=2 时，两张卡的 Worker 为什么都要经历“构造模型、加载权重、profiling、初始化 KV Cache”四个阶段？哪些结果必须在各 rank 之间保持一致？

## 二、从 Hugging Face 配置构造模型

5. 一个 Hugging Face 模型目录中，`config.json`、`model.safetensors.index.json` 和多个 `model-xxxxx-of-xxxxx.safetensors` 分别描述什么？
6. `config.json` 中的 `architectures=["Qwen2ForCausalLM"]` 如何最终决定 Python 中实例化哪个类？如果 registry 不支持这个 architecture，应该在哪一层报错？
7. 以 Qwen2 为例，从外到内描述 `Qwen2ForCausalLM -> Qwen2Model -> Qwen2DecoderLayer` 的模块树。embedding、attention、MLP、RMSNorm、lm_head 分别位于哪里？
8. 为什么在加载任何权重之前，必须先完整构造 `nn.Module` 参数树？`named_parameters()` 在匹配 checkpoint 时扮演什么角色？
9. `tie_word_embeddings=true` 时，`lm_head.weight` 和 `embed_tokens.weight` 在模型实例中应是什么关系？为什么 checkpoint 里可能没有单独的 `lm_head.weight`？
10. 如果 `hidden_size=3584`、`num_attention_heads=28`、`num_key_value_heads=4`，请算出 `head_dim`，并说出一层未做 TP 切分的 K/V cache 每个 token 需要保存多少个标量。
11. 为什么 vLLM 的 Qwen2 会把 HF 的 `q_proj/k_proj/v_proj` 映射到一个 packed `qkv_proj`，把 `gate_proj/up_proj` 映射到 `gate_up_proj`？这会给权重加载器增加什么工作？
12. TP=2 时，哪些线性层适合按输出维切分，哪些适合按输入维切分？如果本项目当前不实现 TP 权重切分，应该明确拒绝什么配置，而不是悄悄加载错误权重？

## 三、safetensors 流式权重加载

13. safetensors 相比 `torch.load(.bin)` 的关键安全和加载特性是什么？“mmap 加载”是否等于数据完全不经过 CPU DDR？
14. 解释一次参数加载的数据路径：磁盘、OS page cache、mmap tensor view、模型参数存储、PCIe、GPU HBM 分别处在哪一步。
15. 为什么权重加载器采用 `(name, tensor)` 迭代器，而不是先把整个 checkpoint 读成一个巨大的 `state_dict`？
16. 多 shard safetensors 应按什么顺序遍历？`model.safetensors.index.json` 在过滤和定位参数时有什么价值？
17. 权重名递归匹配时，如何从 `model.layers.3.self_attn.q_proj.weight` 一层层定位到模型子模块？最后一段 `weight` 与前面的 module path 各代表什么？
18. 如果 checkpoint 出现模型没有的参数，和模型出现 checkpoint 未加载的参数，两种情况分别可能意味着什么？加载器应该如何报告？
19. 为什么不能仅凭 tensor 的元素总数相同就执行 `copy_`？至少要校验哪些属性？
20. 权重加载结束后为什么要执行 `model.eval()`？它改变参数吗，还是改变部分模块的运行行为？

## 四、profiling run 与峰值显存

21. 为什么 profiling 必须发生在模型权重已经加载、分布式通信已经初始化之后，但 KV Cache 还没有分配之前？
22. profiling 前的显存 snapshot 中，`free_memory` 和模型已经占用的显存各自代表什么？为什么需要先 `gc.collect()`、`empty_cache()` 和 `reset_peak_memory_stats()`？
23. dummy/profile batch 为什么通常使用 `max_num_batched_tokens` 与 `max_num_seqs`？只用一条长度为 1 的请求会导致什么后果？
24. profiling run 应覆盖哪些临时显存：模型 forward activation、logits/采样临时量、NCCL/通信 buffer、CUDA graph 预留？哪些内存属于模型常驻权重，不应被重复算作 KV Cache？
25. 请解释公式：`available_kv_cache_memory = requested_memory - non_kv_cache_memory - graph_memory`。其中每一项如何获得？
26. `gpu_memory_utilization=0.9` 是“拿当前 free memory 的 90%”，还是“拿设备总显存的 90% 作为引擎预算”？两种算法在 GPU 上已有其他进程时会产生什么差异？
27. 多个 Worker profiling 得到的可用字节数不同，EngineCore 为什么通常必须采用最小值计算公共 `num_blocks`？
28. profiling dummy forward 失败或 OOM 时，为什么不能简单捕获异常后继续使用 CLI 默认 block 数？应该向用户暴露哪些诊断信息？

## 五、从 KV 规格到物理显存

29. `KVCacheSpec`、`KVCacheGroupSpec`、`KVCacheTensor`、`KVCacheConfig` 四者分别回答了什么问题？请区分“单 block 形状”“共享 block table 的层组”“物理显存区域”“全局初始化方案”。
30. 对 full attention，已知 `block_size`、本 rank 的 `num_kv_heads`、`head_size`、KV dtype，如何计算一个 layer 的 page size（单个 block 字节数）？为什么还要乘 2？
31. 已知每层 page size、层数和可用于 KV Cache 的总字节数，如何求 `num_gpu_blocks`？为什么必须为 null block 额外保留至少一个 block？
32. 为什么 Worker 可以先按 `KVCacheTensor.size` 分配一维 `uint8` 原始 buffer，再通过 `view/reshape` 得到带有 block 维度的 K/V cache？这一步会复制数据吗？
33. 请写出一个典型 full-attention KV Cache 的逻辑形状，并指出哪一维由 `block_id` 直接索引。
34. Scheduler/BlockPool 中的 `block_id` 为什么可以直接成为 GPU KV tensor 第一维的下标？CPU 侧是否还需要维护一张“逻辑 id -> 物理 id”转换表？
35. `KVCacheTensor.shared_by` 表示什么？多个 layer name 共享一片底层 buffer，与多个 layer 各自拥有独立 view 有什么区别？
36. 混合模型为什么可能需要多个 KV cache group？不同 group 的 block table、page size、block 数应满足什么约束？
37. TP=2 后，每个 rank 的 `num_kv_heads` 如何变化？这会如何改变单卡 page size、单卡可容纳 block 数和整个调度域采用的 block 数？
38. 初始化完成后，如何证明 CPU `BlockPool` 的 `num_blocks` 与每个 Worker 真实分配的 block 维度完全一致？你会设计哪些断言和日志？

## 六、故障推演与代码审查

39. 模型权重已经占满大部分显存，profiling 算出的 KV Cache 可用字节小于两个 block 时，引擎应该如何失败？错误信息应该包含哪些量？
40. 某层的 KV view reshape 失败，最可能是哪几个量不一致：总 buffer size、page size、block 数、dtype element size、attention backend layout？
41. 如果先构造 Scheduler/BlockPool，再做 Worker profiling，最终 profiling 算出的 block 数与 CLI 占位值不同，会破坏哪些不变量？
42. 代码审查时看到 `torch.zeros(..., dtype=model_dtype)` 直接分配原始 KV buffer，你会追问什么？为什么 vLLM 常用字节 buffer 描述物理区域，再按 KV dtype/view 解释？
43. 如果 profiling run 调用了真实模型 forward，但本阶段尚未实现 InputBatch，你会如何设计一个最小 dummy forward 接口，使它不污染后续 InputBatch 的职责边界？
44. 请用三分钟总结今天这段架构，要求必须包含这五个词：参数树、mmap、峰值显存、page size、block_id。

## 自测标准

- **能复述**：知道类名、方法名与调用顺序。
- **能解释**：说清每层为什么存在、数据和显存在哪里。
- **能计算**：给定模型配置能算 head size、page size、block 数与 tensor shape。
- **能排错**：面对缺权重、shape mismatch、profiling OOM、reshape 失败能定位责任层。
- **能改代码**：不借助 InputBatch，也能完成模型加载、profiling 和 KV Cache 初始化的最小闭环。
