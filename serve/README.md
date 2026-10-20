# Serving the judge

The judge is a plain chat model behind an OpenAI-compatible API. mu's `logprob` judge sends one request per question with `max_tokens: 1`, `temperature: 0`, `logprobs: true`, `top_logprobs: 20` and `chat_template_kwargs: {"enable_thinking": false}`, and reads the answer off the first token.

## vLLM

```bash
vllm serve <merged model dir or HF repo> --served-model-name mu-judge --port 8000 \
  --max-model-len 8192 --max-logprobs 20 --enable-prefix-caching
```

Measured with vLLM 0.32.0 on a Colab A100 40 GB, serving a merged LoRA model from a smoke run, started without `--enable-prefix-caching`; one warm request each:

| Question | Prompt tokens | Latency |
| --- | --- | --- |
| tool.admission `kind` | 237 | 43 ms |
| turn.drift `course` | 713 | 52 ms |
| turn.completion `claims_done` | 224 | 40 ms |
| turn.continue `promised` | 407 | 41 ms |
| tool.risk `destructive` | 260 | 41 ms |
| 8 questions at once | | 164 ms wall |

## Nebius AI Cloud

`deploy-nebius.sh` creates a Serverless AI endpoint (vLLM, one L40S, bearer token). Then, in `~/.mu/agent/mu.json`:

```json
{
  "tiers": ["mu-judge"],
  "judges": {
    "mu-judge": {
      "type": "logprob",
      "baseUrl": "https://<endpoint>/v1",
      "model": "mu-judge",
      "apiKeyEnv": "MU_JUDGE_ENDPOINT_TOKEN",
      "body": { "chat_template_kwargs": { "enable_thinking": false } }
    }
  }
}
```

A cascade keeps the small judge first and sends only its uncertain answers on: `"tiers": ["mu-judge", "nemotron"]` asks Nemotron on Token Factory about the rest.
