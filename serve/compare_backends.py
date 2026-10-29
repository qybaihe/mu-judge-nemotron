"""Replay requests the fine-tuned judge answered on CUDA (A100) against a judge server on another backend (MLX on a
Mac) and compare the answers question by question. The cache is mu's judge-request cache of a CUDA run: a folder with
`<name>.requests.jsonl` ({"key", "state", "questions"}) and `<name>.jsonl` ({"key", "response", "latencyMs"}).

    python3 serve/compare_backends.py <cache dir> http://127.0.0.1:8770/v1/systemone <n> <out.jsonl>

Prints, per answer type: how often the two agree on the answer (same choice; for a boolean or a score, the same
side of 0.5 / the same value), the largest and mean difference in probability, and latency on both.
"""

import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

cache_dir, url, n, out = (
    Path(sys.argv[1]),
    sys.argv[2],
    int(sys.argv[3]),
    Path(sys.argv[4]),
)
req = {}
for f in cache_dir.glob("*.requests.jsonl"):
    for line in f.open():
        r = json.loads(line)
        req[r["key"]] = r
cached = {}
for f in cache_dir.glob("*.jsonl"):
    if f.name.endswith(".requests.jsonl"):
        continue
    for line in f.open():
        c = json.loads(line)
        cached[c["key"]] = c
keys = sorted(k for k in req if k in cached)
random.Random(0).shuffle(keys)
keys = keys[:n]


def post(body):
    data = json.dumps(body).encode()
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read()), (time.time() - t) * 1000


# The cache holds answers as mu stored them ({"type": "boolean", "probability"}); the server answers on the wire
# ({"noul": p}, {"score": x}, {"choice", "probabilities"}). Both are read by the reference answer's type.
def probs(a, t):
    if t == "boolean":
        return {"yes": a["probability"] if "probability" in a else a["noul"]}
    if t == "score":  # an expected value on 0-3: compared as a share of the scale
        return {"score": a["score"] / 3}
    return {k: v for k, v in (a.get("probabilities") or {}).items()}


def decision(a, t):
    if t == "boolean":
        return probs(a, t)["yes"] >= 0.5
    if t == "score":
        return round(a["score"])
    return a.get("choice")


stats = {}
lat_mlx, lat_cuda = [], []
with out.open("w") as fo:
    for i, k in enumerate(keys):
        r = req[k]
        got, ms = post({"state": r["state"], "questions": r["questions"]})
        lat_mlx.append(got.get("latency_ms", ms))
        lat_cuda.append(cached[k].get("latencyMs"))
        ref = cached[k]["response"]["answers"]
        for qid, a in ref.items():
            b = got["answers"].get(qid)
            t = a.get("type", "?")
            s = stats.setdefault(t, {"n": 0, "agree": 0, "diffs": []})
            s["n"] += 1
            if b is None:
                continue
            s["agree"] += decision(a, t) == decision(b, t)
            pa, pb = probs(a, t), probs(b, t)
            d = max((abs(pa[x] - pb.get(x, 0.0)) for x in pa), default=0.0)
            s["diffs"].append(d)
            fo.write(
                json.dumps(
                    {
                        "key": k[:12],
                        "q": qid,
                        "type": t,
                        "cuda": decision(a, t),
                        "mlx": decision(b, t),
                        "maxdiff": round(d, 4),
                    }
                )
                + "\n"
            )
        if (i + 1) % 25 == 0:
            print(
                f"{i + 1}/{len(keys)} requests, mlx p50 {statistics.median(lat_mlx):.0f} ms",
                flush=True,
            )

print(f"\n{len(keys)} requests")
tot_n = tot_a = 0
alld = []
for t, s in sorted(stats.items()):
    d = s["diffs"]
    alld += d
    tot_n += s["n"]
    tot_a += s["agree"]
    print(
        f"{t:10} {s['n']:5} questions  same answer {s['agree'] / s['n']:.2%}  "
        f"prob diff mean {statistics.mean(d):.4f} p99 {sorted(d)[int(len(d) * 0.99) - 1]:.4f} max {max(d):.4f}"
    )
print(
    f"{'all':10} {tot_n:5} questions  same answer {tot_a / tot_n:.2%}  prob diff mean {statistics.mean(alld):.4f} max {max(alld):.4f}"
)
q = lambda xs, p: sorted(xs)[int(len(xs) * p)]
lc = [x for x in lat_cuda if x is not None]
print(
    f"latency per request: Mac MLX p50 {q(lat_mlx, 0.5):.0f} ms p90 {q(lat_mlx, 0.9):.0f} ms; A100 CUDA p50 {q(lc, 0.5):.0f} ms p90 {q(lc, 0.9):.0f} ms"
)
