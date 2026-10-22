"""Judge requests from your own mu sessions: every decision mu recorded with `"recordState": true` in
~/.mu/agent/mu.json carries the state and the questions as the judge received them (secrets redacted). This turns
them into request rows for label_tf.py.

    python pipeline/rows_from_sessions.py ~/.mu/agent/sessions --out data/requests/mine.jsonl

Rows: {"key", "decision", "source": "sessions", "state", "questions"}; a request asked twice is kept once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", type=Path, help="mu's sessions folder, searched recursively for *.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seen: set[str] = set()
    counts = {"files": 0, "decisions": 0, "without state": 0, "rows": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for path in sorted(args.sessions.rglob("*.jsonl")):
            counts["files"] += 1
            for line in path.open(errors="replace"):
                if '"kyrn.decision"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "custom" or entry.get("customType") != "kyrn.decision":
                    continue
                record = entry.get("data") or {}
                counts["decisions"] += 1
                if record.get("state") is None or not record.get("questions"):
                    counts["without state"] += 1
                    continue
                request = [record["state"], record["questions"]]
                key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps({"key": key, "decision": record["specId"], "source": "sessions",
                                      "state": record["state"], "questions": record["questions"]},
                                     ensure_ascii=False) + "\n")
                counts["rows"] += 1
    print(counts, "->", args.out)
    if counts["decisions"] and not counts["rows"]:
        print('No decision carries its request: set "recordState": true in ~/.mu/agent/mu.json and work as usual.')


if __name__ == "__main__":
    main()
