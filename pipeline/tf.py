"""Token Factory chat completions with the spend ledger. Reads NEBIUS_API_KEY from the environment or from
~/.config/nebius/token-factory.env; refuses to call once the ledger reaches the budget."""

import json
import os
import urllib.request
from pathlib import Path

from spend import BUDGET_USD, record, total

BASE = "https://api.tokenfactory.nebius.com/v1"


def api_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY", "")
    env = Path.home() / ".config/nebius/token-factory.env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("NEBIUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("NEBIUS_API_KEY is not set")
    return key


def chat(tag: str, model: str, messages: list, timeout: float = 120, **extra) -> dict:
    if total() >= BUDGET_USD:
        raise SystemExit(f"budget of ${BUDGET_USD} reached; not calling")
    body = json.dumps({"model": model, "messages": messages, **extra}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    record(tag, model, out.get("usage"))
    return out
