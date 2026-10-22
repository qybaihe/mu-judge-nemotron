"""Token Factory spend ledger: every paid call appends one row to data/spend.jsonl; `python pipeline/spend.py` sums it.

Prices are USD per million tokens (models.dev, 2026-10-22). A model missing here is charged at the most expensive
price on the list, so the ledger never under-counts.

    from spend import record
    record("pilot", model, usage)            # usage = the response's "usage" object

    python pipeline/spend.py                 # totals by tag and model, and what is left of the budget
    python pipeline/spend.py --dump runs/x/completions.jsonl --tag eval:x   # book a file of raw completions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "data" / "spend.jsonl"
BUDGET_USD = 50.0
PRICES = {
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.0, 3.0),
    "nvidia/nemotron-3-super-120b-a12b": (0.3, 0.9),
    "nvidia/Nemotron-3.5-Super-VL": (0.3, 0.9),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
}
WORST = max(PRICES.values())


def cost(model: str, usage: dict) -> float:
    price_in, price_out = PRICES.get(model, WORST)
    return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6


def record(tag: str, model: str, usage: dict | None) -> float:
    usage = usage or {}
    usd = cost(model, usage)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": tag,
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "usd": round(usd, 6),
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return usd


def rows():
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line]


def total() -> float:
    return sum(r["usd"] for r in rows())


def add_dump(path: Path, tag: str) -> None:
    """One ledger row per model for a file of raw completions, one JSON object per line with "model" and "usage"."""
    sums = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    for line in path.read_text().splitlines():
        d = json.loads(line)
        u = d.get("usage") or {}
        s = sums[d["model"]]
        s["prompt_tokens"] += u.get("prompt_tokens", 0)
        s["completion_tokens"] += u.get("completion_tokens", 0)
        s["calls"] += 1
    for model, s in sums.items():
        usd = record(f"{tag} ({s['calls']} calls)", model, s)
        print(f"{tag}: {model} {s['calls']} calls, {s['prompt_tokens']:,} in / {s['completion_tokens']:,} out = ${usd:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--tag")
    args = ap.parse_args()
    if args.dump:
        add_dump(args.dump, args.tag or args.dump.parent.name)
    by = defaultdict(float)
    for r in rows():
        by[(r["tag"].split(" (")[0], r["model"])] += r["usd"]
    for (tag, model), usd in sorted(by.items()):
        print(f"  {usd:9.4f}  {tag:40s} {model}")
    spent = total()
    print(f"spent ${spent:.4f} of ${BUDGET_USD:.2f}; left ${BUDGET_USD - spent:.4f}")
    if spent > BUDGET_USD:
        sys.exit(1)


if __name__ == "__main__":
    main()
