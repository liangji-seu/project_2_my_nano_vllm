#!/usr/bin/env bash
set -u

MY_VLLM_URL="${MY_VLLM_URL:-http://127.0.0.1:8010}"
MY_VLLM_MODEL="${MY_VLLM_MODEL:-qwen2.5-0.5b}"
MAX_TOKENS="1024"

send_prompt() {
  local prompt="$1"
  local output curl_status http_status body

  output=$(
    python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[3]}],"max_tokens":int(sys.argv[2]),"stream":False},ensure_ascii=False))' \
      "$MY_VLLM_MODEL" "$MAX_TOKENS" "$prompt" |
      curl --silent --show-error --connect-timeout 5 --max-time 180 \
        --header "Content-Type: application/json" \
        --write-out $'\n%{http_code}' \
        --data-binary @- "${MY_VLLM_URL%/}/v1/chat/completions"
  )
  curl_status=$?

  if (( curl_status != 0 )); then
    printf 'Request failed: curl exited with status %s.\n' "$curl_status" >&2
    if [[ -n "$output" ]]; then
      printf '%s\n' "$output"
    fi
    return 1
  fi

  http_status="${output##*$'\n'}"
  if [[ "$output" == *$'\n'* ]]; then
    body="${output%$'\n'*}"
  else
    body=""
  fi

  printf '%s\n' "$body"
  if [[ ! "$http_status" =~ ^2[0-9][0-9]$ ]]; then
    printf 'Request failed: HTTP %s.\n' "$http_status" >&2
    return 1
  fi
}

if (( $# > 0 )); then
  prompt="$*"
  send_prompt "$prompt"
  exit $?
fi

if [[ ! -t 0 ]]; then
  prompt="$(cat)"
  if [[ -z "$prompt" ]]; then
    exit 0
  fi
  send_prompt "$prompt"
  exit $?
fi

printf 'Endpoint: %s\nModel: %s\nMax output tokens: %s\n' \
  "$MY_VLLM_URL" "$MY_VLLM_MODEL" "$MAX_TOKENS"
printf 'Each prompt is an independent one-turn request; conversation history is not retained.\n'
printf 'Enter /exit or /quit to leave.\n'
while true; do
  if ! IFS= read -r -p 'Prompt> ' prompt; then
    printf '\n'
    break
  fi
  case "$prompt" in
    /exit|/quit) break ;;
    "") continue ;;
  esac
  if ! send_prompt "$prompt"; then
    printf 'Request failed; you can try another prompt or /exit.\n' >&2
  fi
done
