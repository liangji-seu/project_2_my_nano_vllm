 curl -s --noproxy '*' http://127.0.0.1:13311/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "/home/liangji/huggingface/Qwen2.5-0.5B-Instruct",
      "messages": [
        {"role": "user", "content": "介绍一下你自己"}
      ],
      "max_tokens": 64,
      "temperature": 0.7
    }' | jq .
