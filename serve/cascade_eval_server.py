"""Measure a cascade the way mu runs one ("tiers": ["nemotron-local", "nemotron"]) without serving the student again:
the student's answers are replayed from a judge-request cache, and the questions mu would call uncertain go on to a
Nemotron model on Token Factory through mu's LLM judge prompt. Speaks System One, so any harness that talks to mu's
judges can score it. For evaluation only; in use, mu runs the cascade itself.

    python serve/cascade_eval_server.py --cache runs/_cache-student --escalate nvidia/Nemotron-3-Ultra-550b-a55b \
        --port 8772 [--think]

--cache holds <provider>.requests.jsonl ({"key", "state", "questions"}) and <provider>.jsonl ({"key", "response"}),
the student's answers as mu's Answer objects. GET /stats reports how many questions were escalated.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from judge_format import from_wire  # noqa: E402
from label_tf import SYSTEM_PROMPT, extract_json, to_answer, user_message  # noqa: E402
from spend import record  # noqa: E402
from tf import BASE, api_key  # noqa: E402

# mu's defaults (packages/kyrn-judge/src/policy.ts, cascade.ts): a yes/no answer strictly between 0.2 and 0.8, or a
# choice below 0.6, is uncertain and goes to the next tier.
NO, YES, MIN_CHOICE = 0.2, 0.8, 0.6


def norm(state, questions) -> str:
    return json.dumps([state, {k: from_wire(q) for k, q in questions.items()}], sort_keys=True, ensure_ascii=False)


def uncertain(answer: dict) -> bool:
    if answer["type"] == "boolean":
        return NO < answer["probability"] < YES
    if answer["type"] == "choice":
        p = (answer.get("probabilities") or {}).get(answer["choice"])
        return p is not None and p < MIN_CHOICE
    return False


def to_wire(answer: dict) -> dict:
    if answer["type"] == "boolean":
        return {"noul": answer["probability"]}
    if answer["type"] == "score":
        return {"score": answer["score"]}
    return {"choice": answer["choice"], "probabilities": answer.get("probabilities") or {answer["choice"]: 1.0}}


class Cascade:
    def __init__(self, cache: Path, model: str, think: bool):
        self.answers: dict[str, dict] = {}
        for requests in glob.glob(str(cache / "*.requests.jsonl")):
            responses = {}
            for line in open(requests.replace(".requests.jsonl", ".jsonl")):
                row = json.loads(line)
                responses[row["key"]] = row["response"]["answers"]
            for line in open(requests):
                row = json.loads(line)
                if row["key"] in responses:
                    self.answers[norm(row["state"], row["questions"])] = responses[row["key"]]
        self.model, self.key = model, api_key()
        self.extra = ({"temperature": 0, "max_tokens": 6000, "reasoning_effort": "high"} if think else
                      {"temperature": 0, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False}})
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "questions": 0, "escalated": 0, "missing": 0, "escalation_failed": 0}

    def escalate(self, state, questions: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                  {"role": "user", "content": user_message(state, questions)}], **self.extra}
        req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        record(f"cascade:{self.model.split('/')[-1]}", self.model, out.get("usage"))
        parsed = extract_json((out["choices"][0].get("message") or {}).get("content") or "")
        raw = parsed.get("answers") if isinstance(parsed, dict) else None
        return {qid: to_answer(q, (raw or {}).get(qid)) for qid, q in questions.items()}

    def answer(self, body: dict) -> dict:
        state, questions = body["state"], {k: from_wire(q) for k, q in body["questions"].items()}
        local = self.answers.get(norm(state, questions))
        if local is None:
            with self.lock:
                self.stats["missing"] += 1
            raise LookupError("no student answer recorded for this request")
        unsure = {qid: questions[qid] for qid, a in local.items() if qid in questions and uncertain(a)}
        final = dict(local)
        if unsure:
            try:
                for qid, a in self.escalate(state, unsure).items():
                    if a is not None:
                        final[qid] = a
            except Exception:  # noqa: BLE001 - mu keeps the uncertain answer when the next tier fails
                with self.lock:
                    self.stats["escalation_failed"] += 1
        with self.lock:
            self.stats["requests"] += 1
            self.stats["questions"] += len(questions)
            self.stats["escalated"] += len(unsure)
        return {"model": "cascade", "answers": {qid: to_wire(a) for qid, a in final.items() if qid in questions}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--escalate", default="nvidia/Nemotron-3-Ultra-550b-a55b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--port", type=int, default=8772)
    args = ap.parse_args()
    cascade = Cascade(args.cache, args.escalate, args.think)
    print(f"{len(cascade.answers)} student answers loaded", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, cascade.stats if self.path.startswith("/stats") else {"status": "ok"})

        def do_POST(self):
            try:
                self.send(200, cascade.answer(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))))
            except LookupError as e:
                self.send(404, {"error": {"message": str(e)}})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
