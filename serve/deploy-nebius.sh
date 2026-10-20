#!/usr/bin/env bash
# Serve the trained judge on a Nebius AI Cloud Serverless endpoint: vLLM behind an OpenAI-compatible API,
# one L40S GPU, bearer-token auth. Follows https://docs.nebius.com/serverless/tutorials/deploy-model.
#
#   MODEL_ID=qybaihe/mu-judge-nemotron-nano-4b ./serve/deploy-nebius.sh
#
# Needs the `nebius` CLI configured for a project in eu-north1, and quota for one L40S VM. Prints the token
# and the settings to give mu:
#   ~/.mu/agent/mu.json  {"tiers": ["mu-judge"], "judges": {"mu-judge": {"type": "logprob",
#       "baseUrl": "https://<endpoint>/v1", "model": "mu-judge", "apiKeyEnv": "MU_JUDGE_ENDPOINT_TOKEN",
#       "body": {"chat_template_kwargs": {"enable_thinking": false}}}}}
set -euo pipefail

MODEL_ID="${MODEL_ID:?set MODEL_ID to the Hugging Face repository of the merged model}"
NAME="${NAME:-mu-judge}"
# vLLM 0.32.0 is the version the judge was verified with (Nemotron-H, logprobs, chat_template_kwargs).
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
TOKEN="${TOKEN:-$(openssl rand -hex 32)}"
SUBNET_ID="${SUBNET_ID:-$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')}"

# --max-logprobs 20: the judge reads the top 20 alternatives of the first token.
# --enable-prefix-caching: the questions of one state share everything up to the question.
nebius ai endpoint create \
  --name "$NAME" \
  --image "$IMAGE" \
  --container-command "python3 -m vllm.entrypoints.openai.api_server" \
  --args "--model $MODEL_ID --served-model-name mu-judge --host 0.0.0.0 --port 8000 --max-model-len 8192 --max-logprobs 20 --enable-prefix-caching" \
  --platform gpu-l40s-a \
  --preset 1gpu-8vcpu-32gb \
  --public \
  --container-port 8000 \
  --auth token \
  --token "$TOKEN" \
  --shm-size 16Gi \
  --subnet-id "$SUBNET_ID"

echo "MU_JUDGE_ENDPOINT_TOKEN=$TOKEN"
echo "Wait for Running: nebius ai endpoint list; then point mu's judge at https://<endpoint url>/v1"
