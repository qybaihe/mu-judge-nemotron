"""Join exported cases with teacher labels and split them into train / val / test.

The split is by repository, so no repository the student trained on appears in its test set.
Questions where the teacher put under --min-mass of its first token on the labels are dropped:
there the teacher did not answer in the format, and its distribution says little.

Each (decision point, question, teacher's pick) keeps at most --cap examples, so a question
whose answer is nearly always "no" does not drown the rarer answers.

  uv run pipeline/build_dataset.py --cases data/cases/x.jsonl --labels data/labels/x.ultra.jsonl --out data/dataset
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict


def split_of(repo, val, test):
    bucket = int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if bucket < test else "val" if bucket < test + val else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-mass", type=float, default=0.5)
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--val", type=float, default=0.05)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    labels = {}
    for path in args.labels:
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if "error" not in record:
                    labels[record["case"]] = record

    buckets = defaultdict(list)
    dropped = Counter()
    for path in args.cases:
        with open(path, encoding="utf-8") as source:
            for line in source:
                case = json.loads(line)
                record = labels.get(case["case"])
                if record is None:
                    dropped["unlabeled"] += 1
                    continue
                if record["mass"] < args.min_mass:
                    dropped["low mass"] += 1
                    continue
                order = list(case["labels"].keys())
                target = [record["probs"][label] for label in order]
                pick = order[max(range(len(order)), key=lambda i: target[i])]
                repo = (case.get("facts") or {}).get("repo") or case["transcript"]
                example = {
                    "case": case["case"],
                    "group": case["group"],
                    "spec": case["spec"],
                    "question": case["question"],
                    "type": case["type"],
                    "messages": case["messages"],
                    "labels": order,
                    "target": target,
                    "teacher": record["model"],
                    "teacher_mass": record["mass"],
                    "split": split_of(repo, args.val, args.test),
                }
                buckets[(case["spec"], case["question"], pick)].append(example)

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    files = {name: open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") for name in ("train", "val", "test")}
    counts = Counter()
    picks = Counter()
    for key, examples in sorted(buckets.items()):
        rng.shuffle(examples)
        # The cap applies to training only: val and test keep the teacher's natural distribution.
        kept_train = 0
        for example in examples:
            if example["split"] == "train":
                if kept_train >= args.cap:
                    continue
                kept_train += 1
            files[example["split"]].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[(example["split"], example["spec"])] += 1
            picks[key] += example["split"] == "train"
    for file in files.values():
        file.close()

    print("dropped:", dict(dropped))
    for (split, spec), count in sorted(counts.items()):
        print(f"  {split:5} {spec}: {count}")
    print("training examples per (point, question, teacher pick):")
    for (spec, question, pick), count in sorted(picks.items()):
        print(f"  {spec} {question} -> {pick}: {count}")


if __name__ == "__main__":
    main()
