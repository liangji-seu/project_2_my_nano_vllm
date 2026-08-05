import os
from vllm import LLM, SamplingParams

prompts = [
    "给我介绍一下东南大学",
    "你知道北京烤鸭吗",
    "中国的面积是多少",
    "你知道自嘲熊吗",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_path = os.path.expanduser("~/huggingface/Qwen2.5-0.5B-Instruct/")
llm = LLM(model=model_path,gpu_memory_utilization=0.25)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt}\n Generated text : {generated_text}")