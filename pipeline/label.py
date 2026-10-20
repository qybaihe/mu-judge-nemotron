"""Label exported cases with a teacher model's first-token distribution over the labels.

The request is the one mu's logprob judge sends (packages/kyrn-judge/src/providers/logprob.ts):
the case's own messages, max_tokens 1, temperature 0, top_logprobs 20. The teacher's soft
distribution, not just its pick, is what the student is trained to match.

  NEBIUS_API_KEY=... uv run pipeline/label.py data/cases/x.jsonl data/labels/x.ultra.jsonl \
      --model nvidia/Nemotron-3-Ultra-550b-a55b --body '{"reasoning_effort": "none"}'

Resumable: cases already in the output file are skipped. One output line per case:
  {"case", "model", "probs": {label: p}, "mass", "top": first token, "usage": {...}} or {"case", "error"}.
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time

import httpx

TOP_LOGPROBS = 20


def label_distribution(top_logprobs, labels):
    """Same matching as labelDistribution in logprob.ts: yes/no ignore case, other labels match exactly."""
    sums = {label: 0.0 for label in labels}
    for entry in top_logprobs:
        token, logprob = entry.get("token"), entry.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            continue
        token = token.strip()
        for label in labels:
            if (label in ("yes", "no") and token.lower() == label) or (label not in ("yes", "no") and token == label):
                sums[label] += math.exp(logprob)
                break
    mass = sum(sums.values())
    return {label: (value / mass if mass > 0 else 0.0) for label, value in sums.items()}, mass


async def ask(client, args, extra, case):
    body = {
        **extra,
        "model": args.model,
        "messages": case["messages"],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }
    for attempt in range(6):
        try:
            started = time.perf_counter()
            response = await client.post("/chat/completions", json=body)
            elapsed = round((time.perf_counter() - started) * 1000)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
            if response.status_code != 200:
                return {"case": case["case"], "error": f"HTTP {response.status_code}: {response.text[:300]}"}
            data = response.json()
            content = data["choices"][0].get("logprobs", {}).get("content") or []
            if not content:
                return {"case": case["case"], "error": "no logprobs"}
            top = content[0].get("top_logprobs") or []
            # Exported cases map label -> answer; dataset rows list the labels in order. Either way, the labels.
            probs, mass = label_distribution(top, list(case["labels"]))
            return {
                "case": case["case"],
                "model": args.model,
                "probs": probs,
                "mass": round(mass, 4),
                "top": top[0].get("token") if top else None,
                "usage": data.get("usage"),
                "ms": elapsed,
            }
        except (httpx.HTTPError, KeyError, ValueError) as error:
            if attempt == 5:
                return {"case": case["case"], "error": f"{type(error).__name__}: {str(error)[:200]}"}
            await asyncio.sleep(min(30, 2**attempt + random.random()))


async def run(args):
    key = os.environ.get(args.key_env)
    if not key:
        sys.exit(f"{args.key_env} is not set")
    extra = json.loads(args.body) if args.body else {}
    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as existing:
            for line in existing:
                record = json.loads(line)
                if "error" not in record:
                    done.add(record["case"])
    cases = []
    with open(args.cases, encoding="utf-8") as source:
        for line in source:
            case = json.loads(line)
            if case["case"] not in done and (not args.specs or case["spec"] in args.specs):
                cases.append(case)
    if args.limit:
        random.Random(args.seed).shuffle(cases)
        cases = cases[: args.limit]
    print(f"{len(done)} already labeled, {len(cases)} to go", flush=True)

    queue = asyncio.Queue()
    for case in cases:
        queue.put_nowait(case)
    stats = {"ok": 0, "error": 0, "prompt_tokens": 0, "low_mass": 0}
    started = time.time()
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=120, limits=limits) as client:
        with open(args.output, "a", encoding="utf-8") as out:

            async def worker():
                while not queue.empty():
                    case = queue.get_nowait()
                    record = await ask(client, args, extra, case)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if "error" in record:
                        stats["error"] += 1
                    else:
                        stats["ok"] += 1
                        stats["prompt_tokens"] += (record.get("usage") or {}).get("prompt_tokens") or 0
                        stats["low_mass"] += record["mass"] < 0.5
                    total = stats["ok"] + stats["error"]
                    if total % 500 == 0:
                        out.flush()
                        rate = total / max(1e-9, time.time() - started)
                        print(f"{total}/{len(cases)} {rate:.1f}/s {json.dumps(stats)}", flush=True)

            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    print(f"done {json.dumps(stats)} in {time.time() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.tokenfactory.nebius.com/v1")
    parser.add_argument("--key-env", default="NEBIUS_API_KEY")
    parser.add_argument("--body", help="extra request fields as JSON")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--specs", nargs="*", help="only these decision points")
    parser.add_argument("--limit", type=int, help="label a random sample of this size")
    parser.add_argument("--seed", type=int, default=7)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
