# mu-judge-nemotron

A local judge for the [mu](https://github.com/qybaihe/mu) coding agent: **NVIDIA Nemotron 3 Ultra** on **Nebius Token Factory** teaches **NVIDIA Nemotron 3 Nano 4B**, which then judges on your machine.

mu asks a judge model bounded questions at 40 decision points in its agent loop: is this command safe to run for this task, is this chunk of test output worth reading, did the agent stop before doing what it said, has it drifted from the goal. A judge must be fast, cheap and calibrated, not eloquent. Nemotron 3 Ultra makes the best judge of the Nemotron models on Token Factory, but every decision then costs a network round trip and tokens. This repository distils Ultra's judgment into Nemotron 3 Nano 4B, which reads each answer off a single token. In mu, `"tiers": ["nemotron-local", "nemotron"]` lets the small judge settle what it is sure of and sends only the rest to Ultra.

## How it works

```
judge requests from mu's sessions     ──▶  Nemotron 3 Ultra on Token Factory answers them          (pipeline/label_tf.py)
({state, questions}, as mu sends them)      through mu's own LLM-judge prompt; thinking on for the
                                            decision points where Ultra is weakest
                                                         │
                                                         ▼
one example per question: the prompt, the answer labels (yes/no, A–I, 0–3), Ultra's distribution (build_dataset.py)
                                                         │
                                                         ▼
Nemotron 3 Nano 4B + LoRA learns to match it on the first answer token                  (pipeline/train.py)
                                                         │
                                                         ▼
a System One server, the protocol mu's judges speak: all questions of a request         (serve/systemone_server.py)
in one forward pass, on CUDA (or Apple silicon, experimental)
```

- **One token per answer.** Every label is a single token in the Nemotron tokenizer, so the student's probability for each answer is read directly off its first-token distribution. Nothing is generated or parsed, and the format cannot break.
- **Soft labels.** The student matches the teacher's whole distribution, not just its pick, so its probabilities stay usable for mu's three-zone policies (no ≤ 0.2, unsure, yes ≥ 0.8) and for the cascade.
- **Thinking where it pays.** Ultra's labels for the 8 decision points where it was weakest come with `reasoning_effort: "high"`; with thinking on those 8, its pooled dev score rose from 47.3 to 48.7 on real requests and from 47.7 to 56.0 on constructed ones. The rest are labelled with thinking off, at about half the cost per request.
- **Held-out evaluation.** No request that appears in the evaluation sets, and no coding session they come from, is used in training.

## Results

Scores are the share of the cost of mu's no-judge fallbacks that the judge removes, measured by what mu then *does* on each held-out request (0 = no better than no judge, 100 = every decision right); 1,353 dev requests (541 real, 700 constructed, 112 not shown here).

| Judge | Runs | Real requests | Constructed | Latency |
| --- | --- | --- | --- | --- |
| Nemotron 3 Ultra, thinking off (the teacher) | Token Factory | 47.3 | 47.7 | 0.9 s p50 |
| Nemotron 3 Nano 4B, untuned, same prompts | local | −3.6 | 13.0 | |
| **Nemotron 3 Nano 4B, distilled (this repo)** | local | [[ ]] | [[ ]] | [[ ]] |
| Distilled Nano → Ultra cascade | local + Token Factory | [[ ]] | [[ ]] | [[ ]]% of questions sent to Ultra |
| Nemotron 3 Super, thinking off | Token Factory | 2.7 | −12.7 | |
| Nemotron 3.5 Lightning, first-token logprobs | Token Factory | 15.4 | −12.5 | |

## Use it with mu

```bash
uv pip install -e '.[train]'        # NVIDIA GPU; tested on an A100 40 GB
python serve/systemone_server.py --model <student model folder> --port 8770
```

On Apple silicon, install `'.[mlx]'` and add `--backend mlx`. This backend is experimental: the scores above come from the CUDA backend, and the MLX one is so far checked only on a random model with the student's shape and tokenizer (`serve/check_mlx_backend.py`: the same first-token distributions as a full forward pass), not yet with the trained weights.

Then in `~/.mu/agent/mu.json`:

```json
{ "tiers": ["nemotron-local", "nemotron"] }
```

`nemotron-local` is mu's built-in name for this server at `http://127.0.0.1:8770`; `nemotron` is Nemotron 3 Ultra on Token Factory (`NEBIUS_API_KEY`), asked only about what the local judge is unsure of. `mu ledger` shows which judge answered each question.

## Reproduce

Everything that calls Token Factory books its token usage in `data/spend.jsonl` and stops at a cap you set; `python pipeline/spend.py` prints the totals.

```bash
export NEBIUS_API_KEY=...

# 0. How each Nemotron on Token Factory answers a one-token question (thinking switches, logprobs): < $0.01
python pipeline/probe_tf.py

# 1. Judge requests. From your own mu sessions: set "recordState": true in ~/.mu/agent/mu.json, work as usual, then
python pipeline/rows_from_sessions.py ~/.mu/agent/sessions --out data/requests/mine.jsonl
#    and/or select a decision-balanced, evaluation-disjoint set from request rows:
python pipeline/build_requests.py --rows 'rows/*.jsonl' --exclude-requests eval-requests.jsonl \
    --exclude-tasks eval-tasks.txt --keep-source capture --total 9500 --out data/requests/train.jsonl

# 2. Teacher labels: thinking off everywhere, then thinking on for the decision points where it pays
python pipeline/label_tf.py --requests data/requests/train.jsonl --out data/labels/ultra.jsonl --max-usd 10
python pipeline/label_tf.py --requests data/requests/train.jsonl --out data/labels/ultra-think.jsonl --think \
    --decisions memory.capture,tool.constraint,board.read,input.preflight,skills.disclosure,capability.disclosure,turn.rewind,tool.approval \
    --max-usd 18

# 3. One example per question; later label files override earlier ones
python pipeline/build_dataset.py --requests data/requests/train.jsonl \
    --labels data/labels/ultra.jsonl --labels data/labels/ultra-think.jsonl --out data/dataset

# 4. LoRA on one GPU (A100 40 GB)
python pipeline/train.py --train data/dataset/train.jsonl --val data/dataset/val.jsonl --out runs/student

# 5. Serve, and measure a cascade without serving the student again
python serve/systemone_server.py --model runs/student/merged --port 8770
python serve/cascade_eval_server.py --cache <judge-request cache of the student> --port 8772
```

`serve/systemone_server.py --backend tf --model <Token Factory model>` asks a Token Factory model the same one-token questions, for zero-shot comparisons in the student's own format.

## What it cost

[[$X]] of Token Factory credit for everything above, including every evaluation of every Nemotron model: [[breakdown]]. Training: [[N]] hours on one Colab A100.

## Data and licenses

- Code: MIT.
- Base model: [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), under the NVIDIA Open Model License; the trained adapter and merged weights inherit it.
- Teacher: NVIDIA Nemotron 3 Ultra on Nebius Token Factory.
- The judge requests come from mu's own coding sessions and from generated scenarios. They are not included: sessions hold their authors' content. `rows_from_sessions.py` builds the same kind of data from your own sessions.
