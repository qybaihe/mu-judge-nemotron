"""Turn teacher-labelled judge requests into student examples: one example per question, with the prompt messages,
the answer labels and the teacher's distribution over them. Requests are split into train and val (all questions of
one request on the same side).

    python pipeline/build_dataset.py --requests data/requests/train.jsonl --labels data/labels/ultra-train.jsonl \
        --out data/dataset [--val 0.03]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from judge_format import labels, messages, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--labels", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.03)
    args = ap.parse_args()

    answers = {}
    for path in args.labels:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "answers" in r:
                answers[r["key"]] = r["answers"]

    args.out.mkdir(parents=True, exist_ok=True)
    files = {split: (args.out / f"{split}.jsonl").open("w") for split in ("train", "val")}
    counts = collections.Counter()
    for line in args.requests.read_text().splitlines():
        req = json.loads(line)
        got = answers.get(req["key"])
        if got is None:
            counts["unlabelled request"] += 1
            continue
        bucket = int(hashlib.sha256(req["key"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "val" if bucket < args.val else "train"
        for qid, question in req["questions"].items():
            example = {
                "key": f"{req['key']}/{qid}",
                "request": req["key"],
                "spec": req["decision"],
                "question": qid,
                "type": question["type"],
                "messages": messages(req["state"], question),
                "labels": labels(question),
                "target": target(question, got[qid]),
            }
            files[split].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split] += 1
    for f in files.values():
        f.close()
    print(dict(counts), "->", args.out)


if __name__ == "__main__":
    main()
