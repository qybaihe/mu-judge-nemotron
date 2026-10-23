"""Serve a fine-tuned judge over System One, the protocol mu's judge providers speak: POST /v1/systemone with
{"model", "state", "questions"}; every question is answered from the probabilities of its label tokens on the first
answer token (judge_format), all questions of a request in one forward pass.

    python serve/systemone_server.py --model /content/run/merged --port 8000                       # CUDA, transformers
    python serve/systemone_server.py --backend mlx --model ./mu-judge-nemotron-mlx --port 8000     # Apple silicon
    python serve/systemone_server.py --backend tf --model nvidia/Nemotron-3_5-Lightning           # Token Factory

Then point mu at it (~/.mu/agent/mu.json):
    {"tiers": ["nemotron-local"], "judges": {"nemotron-local": {"type": "systemone",
        "baseUrl": "http://127.0.0.1:8000/v1/systemone", "model": "mu-judge-nemotron"}}}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from judge_format import from_wire, labels, messages, to_system_one  # noqa: E402


class HFBackend:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()

    def encode(self, msgs) -> list[int]:
        ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return ids if isinstance(ids, list) else ids["input_ids"]

    def label_id(self, label: str) -> int:
        return self.tokenizer.encode(label, add_special_tokens=False)[0]

    def first_token_probs(self, prompts: list[list[int]], max_tokens: int = 24_000) -> list:
        """Left-padded batches of at most max_tokens (rows x width); a batch that runs out of memory is halved."""
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        batches, current = [], []
        for i in order:
            if current and (len(current) + 1) * len(prompts[i]) > max_tokens:
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
        while batches:
            batch = batches.pop()
            try:
                for i, probs in zip(batch, self._forward([prompts[i] for i in batch])):
                    out[i] = probs
            except self.torch.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                half = len(batch) // 2
                batches += [batch[:half], batch[half:]]
        return out

    def _forward(self, chunk: list[list[int]]) -> list:
        torch = self.torch
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, p in enumerate(chunk):
            ids[row, width - len(p) :] = torch.tensor(p)
            mask[row, width - len(p) :] = 1
        with torch.inference_mode():
            logits = self.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), logits_to_keep=1).logits[:, -1]
        return list(torch.softmax(logits.float(), dim=-1).cpu())


class MLXBackend:
    """Apple silicon. The questions of one request share their prompt up to the question (system prompt and STATE):
    that prefix runs once into a cache, and each question runs only its own tokens on a copy of it."""

    def __init__(self, model_path: str):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.mx, self.make_prompt_cache = mx, make_prompt_cache
        self.model, self.tokenizer = load(model_path)

    def encode(self, msgs) -> list[int]:
        return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)

    def label_id(self, label: str) -> int:
        return self.tokenizer.encode(label, add_special_tokens=False)[0]

    def first_token_probs(self, prompts: list[list[int]]) -> list:
        import copy

        mx = self.mx
        shared = len(prompts[0])
        for p in prompts[1:]:
            shared = min(shared, next((i for i, (a, b) in enumerate(zip(prompts[0], p)) if a != b), min(len(p), shared)))
        shared = min(shared, min(len(p) for p in prompts) - 1)  # every question keeps at least one token of its own
        cache = self.make_prompt_cache(self.model)
        if shared > 0:
            self.model(mx.array([prompts[0][:shared]]), cache=cache)
            mx.eval([c.state for c in cache])
        out = []
        for p in prompts:
            own = copy.deepcopy(cache) if len(prompts) > 1 else cache
            logits = self.model(mx.array([p[shared:]]), cache=own)[0, -1].astype(mx.float32)
            out.append(mx.softmax(logits, axis=-1))
        return out


class TokenFactoryBackend:
    """The same per-question prompts sent to a model on Nebius Token Factory, read off the first token's top-20
    logprobs (a label outside them counts as 0). For measuring Nemotron models zero-shot in the 4B judge's format;
    every call is booked in the spend ledger."""

    def __init__(self, model: str):
        from concurrent.futures import ThreadPoolExecutor

        from spend import record
        from tf import BASE, api_key

        self.model, self.base, self.key, self.record = model, BASE, api_key(), record
        self.pool = ThreadPoolExecutor(16)

    def encode(self, msgs):
        return msgs

    def label_id(self, label: str) -> str:
        return label

    def ask(self, msgs) -> dict:
        import urllib.request

        body = {"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.loads(resp.read())
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt + 1)
        self.record(f"systemone-tf:{self.model.split('/')[-1]}", self.model, out.get("usage"))
        probs: dict[str, float] = {}
        for alt in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            token = alt["token"].strip()
            probs[token] = probs.get(token, 0.0) + float(2.718281828459045 ** alt["logprob"])
        return probs

    def first_token_probs(self, prompts) -> list:
        return list(self.pool.map(self.ask, prompts))


class Judge:
    def __init__(self, backend, name: str):
        self.backend = backend
        self.name = name
        self.lock = threading.Lock()
        self.ids: dict[str, int] = {}

    def label_ids(self, question: dict) -> list[int]:
        out = []
        for label in labels(question):
            if label not in self.ids:
                self.ids[label] = self.backend.label_id(label)
            out.append(self.ids[label])
        return out

    def answer(self, body: dict) -> dict:
        state = body.get("state")
        questions = {qid: from_wire(q) for qid, q in (body.get("questions") or {}).items()}
        if state is None or not questions:
            raise ValueError("state and questions are required")
        order = list(questions)
        prompts = [self.backend.encode(messages(state, questions[qid])) for qid in order]
        if isinstance(self.backend, TokenFactoryBackend):
            probs = self.backend.first_token_probs(prompts)
        else:
            with self.lock:  # one model on one device
                probs = self.backend.first_token_probs(prompts)
        answers = {}
        for qid, dist in zip(order, probs):
            ids = self.label_ids(questions[qid])
            label_probs = [float(dist.get(i, 0.0)) if isinstance(dist, dict) else float(dist[i]) for i in ids]
            answers[qid] = to_system_one(questions[qid], label_probs)
            answers[qid]["confidence"] = min(1.0, sum(label_probs))
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": sum(len(p) for p in prompts), "output_tokens": 0}}


def make_handler(judge: Judge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def send(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self.send(200, {"object": "list", "data": [{"id": judge.name, "object": "model"}]})
            elif self.path.rstrip("/") in ("", "/health"):
                self.send(200, {"status": "ok", "model": judge.name})
            else:
                self.send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/systemone", "/systemone"):
                self.send(404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                started = time.time()
                result = judge.answer(body)
                result["latency_ms"] = int((time.time() - started) * 1000)
                self.send(200, result)
            except (ValueError, KeyError, TypeError) as e:
                self.send(422, {"error": {"message": str(e)}})
            except Exception as e:  # noqa: BLE001 - the backend failed: say so instead of dropping the connection
                self.send(502, {"error": {"message": f"backend error: {e}"[:300]}})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["hf", "mlx", "tf"], default="hf", help="tf: --model is a Token Factory model id")
    ap.add_argument("--name", default="mu-judge-nemotron")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    backend = {"hf": HFBackend, "mlx": MLXBackend, "tf": TokenFactoryBackend}[args.backend](args.model)
    judge = Judge(backend, args.name)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(judge))
    print(f"System One judge {args.name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
