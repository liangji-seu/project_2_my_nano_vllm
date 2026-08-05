http_proxy= https_proxy= no_proxy=* \
vllm serve ~/huggingface/Qwen2.5-0.5B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.15 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --port 13311
