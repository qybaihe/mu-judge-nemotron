"""Compare judges on the test split: each candidate's labels against the teacher's.

Any model is evaluated the way it is trained and served: label the test cases with label.py
(Token Factory models, or the student behind vLLM), then compare the files here.

  uv run pipeline/compare.py data/dataset/test.jsonl data/labels/test.nano4b.jsonl data/labels/test.lightning.jsonl

Reported per decision point:
  agree   the candidate's pick equals the teacher's
  verdict for yes/no questions, the same side of mu's three-zone policy (no <= 0.2, unsure, yes >= 0.8)
  brier   squared distance between the two distributions over the labels
  mass    share of the candidate's first token on the labels (format adherence)
  ms      median request latency when the label file has it
"""

import argparse
import json
from collections import defaultdict


def median(values):
    if not values:
        return "-"
    ordered = sorted(values)
    return str(ordered[len(ordered) // 2])


def zone(p_yes):
    return "yes" if p_yes >= 0.8 else "no" if p_yes <= 0.2 else "unsure"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()

    test = {}
    with open(args.test, encoding="utf-8") as source:
        for line in source:
            example = json.loads(line)
            test[example["case"]] = example

    for path in args.candidates:
        labels = {}
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                labels[record["case"]] = record
        per = defaultdict(lambda: defaultdict(float))  # "latencies" holds a list
        model = None
        for case, example in test.items():
            record = labels.get(case)
            for key in ("all", example["spec"]):
                stats = per[key]
                stats["n"] += 1
                if record is None or "error" in record:
                    stats["missing"] += 1
                    continue
                model = record.get("model", model)
                order = example["labels"]
                teacher = example["target"]
                mine = [record["probs"].get(label, 0.0) for label in order]
                stats["answered"] += 1
                stats["agree"] += max(range(len(order)), key=lambda i: mine[i]) == max(range(len(order)), key=lambda i: teacher[i])
                stats["brier"] += sum((a - b) ** 2 for a, b in zip(mine, teacher))
                stats["mass"] += record["mass"]
                if "ms" in record:
                    per[key].setdefault("latencies", []).append(record["ms"])
                if example["type"] == "boolean":
                    yes = order.index("yes")
                    stats["bool"] += 1
                    stats["verdict"] += zone(mine[yes]) == zone(teacher[yes])
        print(f"\n{path} ({model})")
        print(f"{'point':18} {'n':>6} {'missing':>8} {'agree':>7} {'verdict':>8} {'brier':>7} {'mass':>6} {'ms':>6}")
        for key in ["all"] + sorted(k for k in per if k != "all"):
            s = per[key]
            answered = max(1, s["answered"])
            verdict = f"{s['verdict'] / s['bool']:.3f}" if s["bool"] else "-"
            print(
                f"{key:18} {int(s['n']):6} {int(s['missing']):8} {s['agree'] / answered:7.3f} {verdict:>8} "
                f"{s['brier'] / answered:7.3f} {s['mass'] / answered:6.3f} {median(s.get('latencies')):>6}"
            )


if __name__ == "__main__":
    main()
