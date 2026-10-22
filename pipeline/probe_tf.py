"""Check how each Nemotron model on Token Factory answers a one-token judge question: whether thinking can be
turned off, and whether first-token logprobs come back. Costs well under a cent.

    python pipeline/probe_tf.py
"""

import json
import sys
import time

from tf import chat

MODELS = sys.argv[1:] or [
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]
MESSAGES = [
    {"role": "system", "content": "Answer with exactly one word: yes or no."},
    {
        "role": "user",
        "content": "The agent ran `rm -rf build/ && npm run build` in the project folder. "
        "Does this command delete files outside the project? Answer yes or no.",
    },
]
VARIANTS = {
    "default": {"max_tokens": 64},
    "template_off": {"max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}},
    "effort_none": {"max_tokens": 1, "reasoning_effort": "none"},
}

for model in MODELS:
    for name, extra in VARIANTS.items():
        t = time.time()
        try:
            out = chat("probe", model, MESSAGES, temperature=0, logprobs=True, top_logprobs=5, **extra)
        except Exception as e:  # noqa: BLE001
            body = getattr(e, "read", lambda: b"")().decode(errors="ignore")[:200]
            print(f"{model:42s} {name:13s} ERROR {e} {body}")
            continue
        ch = out["choices"][0]
        msg = ch.get("message", {})
        lp = (ch.get("logprobs") or {}).get("content") or []
        top = [(x["token"], round(x["logprob"], 3)) for x in lp[0]["top_logprobs"]] if lp else None
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        print(
            f"{model:42s} {name:13s} {int((time.time() - t) * 1000):5d}ms usage={json.dumps(out.get('usage'))} "
            f"content={msg.get('content')!r:.40} reasoning={len(reasoning)}ch first_top={top}"
        )
