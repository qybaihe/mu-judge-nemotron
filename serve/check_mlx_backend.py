"""Check the MLX backend without the real weights: a small random Nemotron-H with Nemotron 3 Nano 4B's Mamba geometry
and tokenizer must give the same first-token distributions through the server's prefix cache as through a full
forward pass of each prompt, and answer a System One request.

    uv run --with 'mlx-lm>=0.32' python serve/check_mlx_backend.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models import nemotron_h

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import systemone_server as server  # noqa: E402
from judge_format import messages  # noqa: E402

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]


def main() -> None:
    folder = Path(tempfile.mkdtemp())
    for name in FILES:
        (folder / name).write_bytes(Path(hf_hub_download(BASE, name)).read_bytes())
    config = json.loads((folder / "config.json").read_text())
    # Small everywhere except the Mamba heads, groups and state, which decide the kernels mlx-lm runs.
    config.update(hidden_size=256, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                  hybrid_override_pattern="M-M*-M-", num_hidden_layers=7)
    (folder / "config.json").write_text(json.dumps(config))
    model = nemotron_h.Model(nemotron_h.ModelArgs.from_dict(config))
    mx.random.seed(0)
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.1, model.parameters()))
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))

    backend = server.MLXBackend(str(folder))
    state = {"task": "fix the failing slugify test", "recent": ["ran pytest: 1 failed"], "command": "python3 test_slugify.py"}
    questions = {
        "safe": {"type": "noul", "instructions": "Is this command safe to run for this task?"},
        "next": {"type": "choice", "instructions": "What should happen next?",
                 "criteria": {"run": "run it", "ask": "ask the user", "skip": "skip it"}},
        "useful": {"type": "score", "instructions": "How useful is this output?", "criteria": ["none", "some", "much", "all"]},
    }
    prompts = [backend.encode(messages(state, server.from_wire(q))) for q in questions.values()]
    full = [mx.softmax(backend.model(mx.array([p]))[0, -1].astype(mx.float32), axis=-1) for p in prompts]
    worst = 0.0
    for batch in (prompts, prompts[:1], prompts):  # the second full batch: the shared cache must not change
        for got, want in zip(backend.first_token_probs(batch), full):
            worst = max(worst, float(mx.max(mx.abs(got - want))))
    print(f"prompts of {[len(p) for p in prompts]} tokens; largest probability difference, cached vs full: {worst:.2e}")
    assert worst < 1e-3, worst
    answer = server.Judge(backend, "check").answer({"state": state, "questions": questions})
    assert set(answer["answers"]) == set(questions), answer
    print("ok:", json.dumps(answer["answers"])[:300])


if __name__ == "__main__":
    main()
