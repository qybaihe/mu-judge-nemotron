"""Turn nebius/SWE-rebench-openhands-trajectories into transcripts the mu exporter replays.

One JSON object per line:
  {"id", "source", "repo", "resolved", "exit_status", "messages": [...]}
Messages keep the OpenAI shape the dataset already has, minus OpenHands' own system prompt:
  {"role": "user" | "assistant" | "tool", "content": str,
   "tool_calls": [{"id", "name", "arguments": {...}}],   # assistant only
   "tool_call_id": str, "name": str}                      # tool only

Sampling is stratified on `resolved` and capped per repository, so a few large repositories
do not dominate and the judge sees as many finished-but-wrong runs as successful ones.
"""

import argparse
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

SOURCE = "nebius/SWE-rebench-openhands-trajectories"
COLUMNS = ["trajectory_id", "repo", "trajectory", "exit_status", "resolved"]


def convert_message(message):
    role = message["role"]
    if role == "assistant":
        calls = []
        for call in message.get("tool_calls") or []:
            raw = call["function"]["arguments"]
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                arguments = {"raw": raw}
            calls.append({"id": call["id"], "name": call["function"]["name"], "arguments": arguments})
        out = {"role": "assistant", "content": message.get("content") or ""}
        if calls:
            out["tool_calls"] = calls
        return out
    if role == "tool":
        return {
            "role": "tool",
            "content": message.get("content") or "",
            "tool_call_id": message.get("tool_call_id") or "",
            "name": message.get("name") or "",
        }
    return {"role": role, "content": message.get("content") or ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet")
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=3000, help="transcripts to keep")
    parser.add_argument("--per-repo", type=int, default=8, help="at most this many transcripts per repository")
    parser.add_argument("--max-messages", type=int, default=160, help="skip longer runs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reservoirs = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    per_repo = defaultdict(int)
    target = {0: args.count // 2, 1: args.count - args.count // 2}

    parquet = pq.ParquetFile(args.parquet)
    for batch in parquet.iter_batches(batch_size=256, columns=COLUMNS):
        for row in batch.to_pylist():
            resolved = int(row["resolved"] or 0)
            trajectory = row["trajectory"] or []
            if len(trajectory) > args.max_messages or per_repo[row["repo"]] >= args.per_repo:
                continue
            per_repo[row["repo"]] += 1
            seen[resolved] += 1
            # Reservoir sampling keeps a uniform sample of each class without holding the file in memory.
            bucket = reservoirs[resolved]
            if len(bucket) < target[resolved]:
                bucket.append(row)
            else:
                slot = rng.randrange(seen[resolved])
                if slot < target[resolved]:
                    bucket[slot] = row

    with open(args.out, "w", encoding="utf-8") as out:
        for row in reservoirs[0] + reservoirs[1]:
            messages = [convert_message(m) for m in row["trajectory"] if m["role"] != "system"]
            record = {
                "id": row["trajectory_id"],
                "source": SOURCE,
                "repo": row["repo"],
                "resolved": int(row["resolved"] or 0),
                "exit_status": row["exit_status"],
                "messages": messages,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"kept {len(reservoirs[0])} unresolved and {len(reservoirs[1])} resolved of {seen[0] + seen[1]} eligible")


if __name__ == "__main__":
    main()
