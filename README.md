# mu-judge-nemotron

A small, fast judge for the [mu](https://github.com/qybaihe/mu) coding agent, distilled from **NVIDIA Nemotron 3 Ultra** into **NVIDIA Nemotron 3 Nano 4B** on **Nebius Token Factory**.

mu asks a judge dozens of bounded questions in every turn: is this chunk of test output noise, has the agent drifted from the goal, did it stop before doing what it said, did the user ask for this destructive command. Each answer is read as a probability off the judge's first token. A judge must therefore be fast, cheap, and calibrated, not eloquent. This repository turns Nemotron 3 Nano 4B into that judge.

## How it works

```
agent transcripts ──▶ mu replays them through its decision points ──▶ questions, rendered exactly as mu asks them
   (nebius/SWE-rebench-                (kyrn/distill/export-cases.ts)
    openhands-trajectories)                                                  │
                                                                             ▼
                              Nemotron 3 Ultra on Token Factory labels each question:
                              its first-token distribution over the labels (pipeline/label.py)
                                                                             │
                                                                             ▼
               Nemotron 3 Nano 4B + LoRA learns to match those distributions (pipeline/train.py)
                                                                             │
                                                                             ▼
           served behind an OpenAI-compatible API; mu's `logprob` judge reads its first token
```

- **Same prompt in training and serving.** mu's exporter renders every question with `renderLabeledPrompt`, the function its `logprob` judge uses at run time. Every answer is one token: `yes`/`no`, `A`-`Z` for options, `0`-`9` for rubric levels.
- **Soft labels.** The student matches the teacher's whole distribution, so its probabilities stay calibrated for mu's three-zone policies (no ≤ 0.2, unsure, yes ≥ 0.8).
- **Split by repository.** No repository in the test set appears in training.

## Decision points covered

| Point | Asked when | Question |
| --- | --- | --- |
| `tool.admission` | a shell output of 6,000+ characters, per chunk | error, result, progress, warning, passing? |
| `turn.drift` | every 6 tool results | on track, detour, drift or loop? |
| `turn.completion` | the run ends after an edit nothing has verified | does it claim to be done; does it need a check? |
| `turn.continue` | the run ends on a message | did it promise a step and stop; ask for a go-ahead it had? |
| `tool.risk` | a command the rules flag | destructive; did the user ask for it? |

## Reproduce

```bash
# 1. Transcripts (CC-BY-4.0, Nebius)
uv run pipeline/convert_openhands.py trajectories.parquet data/transcripts/openhands-3000.jsonl --count 3000
# 2. Questions, from a mu checkout
node kyrn/distill/export-cases.ts data/transcripts/openhands-3000.jsonl data/cases/openhands-3000.jsonl
# 3. Teacher labels on Nebius Token Factory
NEBIUS_API_KEY=... uv run pipeline/label.py data/cases/openhands-3000.jsonl data/labels/openhands-3000.ultra.jsonl \
    --model nvidia/Nemotron-3-Ultra-550b-a55b
# 4. Dataset, split by repository
uv run pipeline/build_dataset.py --cases data/cases/*.jsonl --labels data/labels/*.jsonl --out data/dataset
# 5. Train on one GPU (A100 40 GB: about 1 s per example)
python pipeline/train.py --train data/dataset/train.jsonl --val data/dataset/val.jsonl --out runs/nano4b
```

## Results

Measured on the held-out test split. Filled in as runs finish.

## Data and licenses

- Code: MIT.
- Transcripts: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), CC-BY-4.0, © Nebius.
- Base model: [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), NVIDIA Nemotron Open Model License. The trained weights are a derivative under the same license.
