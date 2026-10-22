"""Collect the judge requests to label: unique (state, questions) pairs from mu judge-request rows, minus everything
that belongs to an evaluation set, then a budgeted, decision-balanced selection.

    python pipeline/build_requests.py --rows 'rows/*.jsonl' --exclude-requests eval-requests.jsonl \
        --exclude-tasks eval-tasks.txt --keep-source capture-r05 --total 9500 --out data/requests/train.jsonl

--rows            mu judge-request rows ({"decision", "source", "family", "state", "questions", ...}); labels in them
                  are ignored.
--exclude-requests  requests ({"state", "questions"}) of the evaluation sets: a row with the same state is dropped.
--exclude-tasks   one task id per line: a row whose family ends in "/<task>" is dropped (whole coding sessions stay
                  out of training).
--canary          a row containing this string is dropped.
--keep-source     rows of these sources are all kept (real captures); the rest of --total is filled with the other
                  rows, the same number per decision where there are enough.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
from pathlib import Path


def state_key(state) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="glob of row files")
    ap.add_argument("--exclude-requests", type=Path, action="append", default=[])
    ap.add_argument("--exclude-tasks", type=Path)
    ap.add_argument("--canary", action="append", default=[])
    ap.add_argument("--keep-source", action="append", default=[])
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    banned_states = set()
    for f in args.exclude_requests:
        for line in f.read_text().splitlines():
            if line:
                banned_states.add(state_key(json.loads(line)["state"]))
    banned_tasks = set(args.exclude_tasks.read_text().split()) if args.exclude_tasks else set()

    pool: dict[str, dict] = {}
    dropped = collections.Counter()
    for f in sorted(glob.glob(args.rows)):
        for line in open(f):
            if any(c in line for c in args.canary):
                dropped["canary"] += 1
                continue
            r = json.loads(line)
            if r.get("family", "").rsplit("/", 1)[-1] in banned_tasks:
                dropped["eval task"] += 1
                continue
            if state_key(r["state"]) in banned_states:
                dropped["eval request"] += 1
                continue
            key = hashlib.sha256(json.dumps([r["state"], r["questions"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
            if key in pool:
                dropped["duplicate"] += 1
                continue
            pool[key] = {"key": key, "decision": r["decision"], "source": r.get("source"), "family": r.get("family"),
                         "lang": r.get("lang"), "state": r["state"], "questions": r["questions"]}
    print(f"pool {len(pool)} unique requests; dropped {dict(dropped)}")

    rng = random.Random(args.seed)
    kept = [r for r in pool.values() if r["source"] in args.keep_source]
    rest = [r for r in pool.values() if r["source"] not in args.keep_source]
    by_decision = collections.defaultdict(list)
    for r in rest:
        by_decision[r["decision"]].append(r)
    for rows in by_decision.values():
        rng.shuffle(rows)
    budget = max(0, args.total - len(kept))
    chosen = []
    # Round-robin over decisions: each gets the same share until it runs out.
    while budget > 0 and any(by_decision.values()):
        for d in sorted(by_decision):
            if budget == 0:
                break
            if by_decision[d]:
                chosen.append(by_decision[d].pop())
                budget -= 1
    out = kept + chosen
    rng.shuffle(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = collections.Counter(r["decision"] for r in out)
    print(f"selected {len(out)} ({len(kept)} kept by source, {len(chosen)} balanced) over {len(per)} decisions -> {args.out}")
    print("  " + ", ".join(f"{d} {n}" for d, n in sorted(per.items())))


if __name__ == "__main__":
    main()
