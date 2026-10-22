"""Label System One judge requests with a Nemotron teacher on Token Factory, through the same prompt mu's LLM judge
sends (packages/kyrn-judge/src/providers/llm.ts), with thinking off. Resumable: requests already in --out are skipped.
Every call is booked in the spend ledger, and the run stops before the ledger passes --max-usd.

    python pipeline/label_tf.py --requests data/requests/train.jsonl --out data/labels/ultra.jsonl \
        --model nvidia/Nemotron-3-Ultra-550b-a55b --max-usd 12 [--limit N] [--concurrency 12]

Input rows: {"key", "decision", "state", "questions"} (mu's Question objects). Output rows add "answers" (mu's Answer
objects, as llm.ts turns the reply into them), "raw" (the reply), "usage" and "ms"; a reply mu would reject gets
"error" instead of "answers".
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from spend import record, total
from tf import BASE, api_key

SYSTEM_PROMPT = """You are a judgment function inside a coding agent harness. You read STATE and answer typed QUESTIONS about it. You never explain.

Answer formats, by question type:
- boolean: {"p": <number>}  probability in [0,1] that the statement is true. Use the whole range; 0.5 means the state does not say.
- choice:  {"choice": "<one option name>", "p": <number>}  p is the probability that this option is the right one.
- score:   {"score": <number>}  position on the rubric from 0 (first level) to N-1 (last level). Fractions are allowed.

Judge only from STATE. Answer every question id. Reply with exactly one JSON object and nothing else:
{"answers": {"<question id>": {...}}}"""


def render_question(q: dict) -> dict:
    if q["type"] == "choice":
        return {"type": "choice", "instructions": q["instructions"], "options": q["criteria"]}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["instructions"], "levels": q["criteria"]}
    out = {"type": "boolean", "instructions": q["instructions"]}
    if q.get("criteria"):
        out["meaning"] = q["criteria"]
    return out


def user_message(state, questions: dict) -> str:
    # JSON.stringify(x, null, 1): one-space indent, no ASCII escaping.
    s = state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)
    qs = json.dumps({k: render_question(q) for k, q in questions.items()}, indent=1, ensure_ascii=False)
    return f"STATE:\n{s}\n\nQUESTIONS:\n{qs}"


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def to_answer(q: dict, raw) -> dict | None:
    """llm.ts toAnswer: the reply for one question as mu's Answer, or None when mu would reject it."""
    if not isinstance(raw, dict):
        return None
    if q["type"] == "boolean":
        p = raw.get("p")
        return {"type": "boolean", "probability": clamp(p, 0, 1)} if isinstance(p, (int, float)) else None
    if q["type"] == "score":
        s = raw.get("score")
        return {"type": "score", "score": clamp(s, 0, len(q["criteria"]) - 1)} if isinstance(s, (int, float)) else None
    options = list(q["criteria"].keys())
    choice = raw.get("choice")
    if not isinstance(choice, str) or choice not in options:
        return None
    p = raw.get("p")
    picked = clamp(p, 0, 1) if isinstance(p, (int, float)) else 1.0
    rest = (1 - picked) / (len(options) - 1) if len(options) > 1 else 0.0
    return {"type": "choice", "choice": choice, "probabilities": {o: picked if o == choice else rest for o in options}}


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def call(model: str, body_extra: dict, row: dict, tag: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(row["state"], row["questions"])},
        ],
        **body_extra,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
    )
    for attempt in range(6):
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt + 1)
                continue
            return {"key": row["key"], "error": f"http {e.code}: {e.read()[:200]!r}"}
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(2 ** attempt + 1)
                continue
            return {"key": row["key"], "error": f"network: {e}"}
    ms = int((time.time() - t) * 1000)
    usage = out.get("usage") or {}
    record(tag, model, usage)
    text = (out["choices"][0].get("message") or {}).get("content") or ""
    result = {"key": row["key"], "decision": row.get("decision"), "raw": text, "usage": {
        "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}, "ms": ms}
    parsed = extract_json(text)
    answers_raw = parsed.get("answers") if isinstance(parsed, dict) else None
    if not isinstance(answers_raw, dict):
        result["error"] = "no answers object"
        return result
    answers = {}
    for qid, q in row["questions"].items():
        a = to_answer(q, answers_raw.get(qid))
        if a is None:
            result["error"] = f"no usable answer for {qid}"
            return result
        answers[qid] = a
    result["answers"] = answers
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="nvidia/Nemotron-3-Ultra-550b-a55b")
    ap.add_argument("--max-usd", type=float, required=True, help="stop before the ledger total passes this")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--think", action="store_true", help="let the teacher think first (reasoning_effort high)")
    ap.add_argument("--decisions", help="comma-separated decision points to label; default all")
    args = ap.parse_args()

    if args.think:
        # The reasoning comes back in its own field; the content is the JSON answer alone.
        body_extra = {"temperature": 0, "max_tokens": 6000, "reasoning_effort": "high"}
    else:
        body_extra = {"temperature": 0, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False}}
    tag = args.tag or f"label:{args.model.split('/')[-1]}{'-think' if args.think else ''}:{args.requests.stem}"
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if "answers" in r:
                done.add(r["key"])
    rows = [json.loads(l) for l in args.requests.read_text().splitlines() if l]
    if args.decisions:
        wanted = set(args.decisions.split(","))
        rows = [r for r in rows if r["decision"] in wanted]
    todo = [r for r in rows if r["key"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} requests, {len(done)} labelled, {len(todo)} to go; ledger ${total():.4f}, cap ${args.max_usd}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stop = threading.Event()
    counts = {"ok": 0, "error": 0}

    def work(row):
        if stop.is_set():
            return None
        if total() >= args.max_usd:
            stop.set()
            return None
        return call(args.model, body_extra, row, tag)

    with ThreadPoolExecutor(args.concurrency) as pool, args.out.open("a") as f:
        futures = [pool.submit(work, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                continue
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                counts["error" if "error" in res else "ok"] += 1
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} {counts} ledger ${total():.3f}", flush=True)
    print(f"done {counts}; ledger ${total():.4f}" + (" (stopped at the cap)" if stop.is_set() else ""))


if __name__ == "__main__":
    main()
