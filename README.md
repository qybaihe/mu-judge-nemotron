# mu-judge-nemotron

A local judge for the [mu](https://github.com/qybaihe/mu) coding agent: **NVIDIA Nemotron 3 Nano 4B**, fine-tuned to answer mu's judge questions on your machine, and measured against every Nemotron on **Nebius Token Factory**.

mu asks a judge model bounded questions at 40 decision points in its agent loop: is this command safe to run for this task, is this chunk of test output worth reading, did the agent stop before doing what it said, has it drifted from the goal. A judge must be fast, cheap and calibrated, not eloquent. Nemotron 3 Ultra makes the best judge of the Nemotron models on Token Factory, but every decision then costs a network round trip and tokens. This repository fine-tunes Nemotron 3 Nano 4B as mu's judge; it reads each answer off a single token. On 4,838 held-out test requests, scored once, the fine-tuned 4B judge scored above Ultra: 68.2 against 54.7 on real requests from coding sessions, 59.9 against 48.4 on constructed scenarios. In mu it is the built-in judge `nemotron-local`.

**Weights:** [qybaihe/mu-judge-nemotron-nano-4b](https://huggingface.co/qybaihe/mu-judge-nemotron-nano-4b) on Hugging Face (bf16, 8 GB). To use it, see [Use it with mu](#use-it-with-mu).

## How it works

```
judge requests from mu's sessions     ──▶  labels: a probability for each answer                   (pipeline/label_tf.py)
({state, questions}, as mu sends them)      of each question
                                                         │
                                                         ▼
one example per question: the prompt, the answer labels (yes/no, A–I, 0–3), their probabilities (build_dataset.py)
                                                         │
                                                         ▼
Nemotron 3 Nano 4B + LoRA learns to match it on the first answer token                  (pipeline/train.py)
                                                         │
                                                         ▼
a System One server, the protocol mu's judges speak: all questions of a request         (serve/systemone_server.py)
in one forward pass, on CUDA (or Apple silicon, experimental)
```

- **One token per answer.** Every label is a single token in the Nemotron tokenizer, so the judge's probability for each answer is read directly off its first-token distribution. Nothing is generated or parsed, and the format cannot break.
- **Soft labels.** The model learns a whole distribution over each question's labels, not just one pick, so its probabilities stay usable for mu's three-zone policies (no ≤ 0.2, unsure, yes ≥ 0.8) and for the cascade.
- **Held-out evaluation.** No request that appears in the evaluation sets, and no coding session they come from, is used in training.

## Results

Scores are the share of the cost of mu's no-judge fallbacks that the judge removes, measured by what mu then *does* on each held-out request (0 = no better than no judge, 100 = every decision right). Dev: 1,353 requests (541 real, from coding sessions; 700 constructed scenarios; 112 written by mu's author).

| Judge | Runs | Real requests | Constructed | Author-written | Latency |
| --- | --- | --- | --- | --- | --- |
| **Nemotron 3 Nano 4B, fine-tuned (this repo)** | local | **62.3** | 45.0 | 19.6 | 0.07–0.1 s to approve a command (A100) |
| Fine-tuned Nano → Ultra cascade | local + Token Factory | **63.1** | 44.1 | 38.6 | about a fifth of questions go to Ultra |
| Nemotron 3 Ultra, thinking off | Token Factory | 47.3 | 47.7 | 52.3 | 0.9 s p50 |
| Nemotron 3 Nano 4B, untuned, same prompts | local | −3.6 | 13.0 | −11.8 | |
| Nemotron 3 Super, thinking off | Token Factory | 2.7 | −12.7 | 76.6 | |
| Nemotron 3.5 Lightning, first-token logprobs | Token Factory | 15.4 | −12.5 | | |

Test: 4,838 further held-out requests (3,244 real, 1,594 constructed), scored once, after everything above was fixed. 95% bootstrap intervals over coding tasks (real) and scenarios (constructed):

| Judge | Real requests | Constructed |
| --- | --- | --- |
| **Nemotron 3 Nano 4B, fine-tuned (this repo)** | **68.2** (61.0 to 76.0) | **59.9** (51.5 to 68.3) |
| Nemotron 3 Ultra, thinking on 4 decision points | 57.9 (48.7 to 66.6) | 55.9 (44.6 to 66.6) |
| Nemotron 3 Ultra, thinking off | 54.7 (44.6 to 64.0) | 48.4 (35.8 to 59.8) |
| Fine-tuned Nano → Ultra cascade | 20.0 (−74.6 to 69.7) | 55.7 (44.8 to 66.4) |

Paired against Ultra, the 4B judge is ahead by 13.6 on real requests (+8.1 to +20.7) and by 11.5 on constructed ones (+1.5 to +22.6). Against Ultra with thinking on four decision points it is ahead by 10.3 on real requests (+5.0 to +17.1); on constructed ones the interval includes zero (+4.1, −5.0 to +14.1).

What the numbers say:

- **The 4B judge scores above Ultra.** On dev that holds for real requests (62.3 against 47.3) but not for constructed ones (45.0 against 47.7); on the larger test split it holds for both. On real requests it is also right more often (89.2% of checks on test against Ultra's 87.2%). Our reading, not a measured cause: mu's cost-weighted policies reward probabilities less extreme than Ultra's usual 0.9 and 0.95.
- **On the 112 author-written requests it is clearly weaker** (19.6 against 52.3 on dev). These are hard, deliberately tricky cases.
- **The cascade did not pay on test.** Sending the questions the 4B judge is unsure of to Ultra helped on the author-written dev requests (38.6) and changed little on the rest of dev, but on real test requests it scored far lower than the 4B judge alone (20.0, with a very wide interval). In a few costly places, such as letting a tool's output into the context (2 mistakes in 520) or keeping a memory, Ultra's confident answers cost more than the 4B judge's doubt, which falls back to what mu does without a judge. The cascade's rule is mu's own (unsure: a yes/no probability between 0.2 and 0.8, or a choice below 0.6) and was not tuned on test. So we recommend the 4B judge on its own.
- The comparison is fair: no request in these sets, and no coding session one comes from, was used in training (checked by task: none of the 29 real dev tasks or 86 real test tasks appears in the training data).

## Use it with mu

```bash
uv pip install -e '.[train]'        # NVIDIA GPU; tested on an A100 40 GB
python serve/systemone_server.py --model qybaihe/mu-judge-nemotron-nano-4b --port 8770
```

The merged weights are on Hugging Face: [qybaihe/mu-judge-nemotron-nano-4b](https://huggingface.co/qybaihe/mu-judge-nemotron-nano-4b) (about 8 GB, downloaded on first start). `--model` also takes a local folder, such as the `runs/judge/merged` that train.py writes (step 4 below).

On Apple silicon, install `'.[mlx]'` and add `--backend mlx`. This backend is experimental: the scores above come from the CUDA backend, and the MLX one is so far checked only on a random model with this model's shape and tokenizer (`serve/check_mlx_backend.py`: the same first-token distributions as a full forward pass), not yet with the trained weights.

Then in `~/.mu/agent/mu.json`:

```json
{ "tiers": ["nemotron-local"] }
```

`nemotron-local` is mu's built-in name for this server at `http://127.0.0.1:8770`. Where it is unsure, mu does what it does without a judge. `mu ledger` lists every verdict with its latency.

To send its unsure questions to Nemotron 3 Ultra on Token Factory (`NEBIUS_API_KEY`), use `"tiers": ["nemotron-local", "nemotron"]`: it helped on the hardest dev requests but did worse on real test requests (see Results).

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

# 2. Labels: a probability for each answer label; a second pass for the decision points listed
python pipeline/label_tf.py --requests data/requests/train.jsonl --out data/labels/labels.jsonl \
    --model <Token Factory model> --max-usd 10
python pipeline/label_tf.py --requests data/requests/train.jsonl --out data/labels/labels-2.jsonl \
    --model <Token Factory model> --think \
    --decisions memory.capture,tool.constraint,board.read,input.preflight,skills.disclosure,capability.disclosure,turn.rewind,tool.approval \
    --max-usd 18

# 3. One example per question; later label files override earlier ones
python pipeline/build_dataset.py --requests data/requests/train.jsonl \
    --labels data/labels/labels.jsonl --labels data/labels/labels-2.jsonl --out data/dataset

# 4. LoRA on one GPU (A100 40 GB)
python pipeline/train.py --train data/dataset/train.jsonl --val data/dataset/val.jsonl --out runs/judge
#    (train.py merges the adapter into runs/judge/merged; from the adapter alone, on any machine:)
python pipeline/merge.py --adapter runs/judge/adapter --out runs/judge/merged

# 5. Serve, and measure a cascade without serving the 4B judge again
python serve/systemone_server.py --model runs/judge/merged --port 8770
python serve/cascade_eval_server.py --cache <judge-request cache of the 4B judge> --port 8772
```

`serve/systemone_server.py --backend tf --model <Token Factory model>` asks a Token Factory model the same one-token questions, for zero-shot comparisons in the 4B judge's own format.

## What it cost

$27.13 of Token Factory credit for everything above: $15.78 for the training labels, $9.10 for measuring every Nemotron as mu's judge, $1.60 for the cascade's calls to Ultra, and $0.64 for end-to-end runs of mu and the demos (the desktop app run included). Training: about 3.1 hours on one Colab A100 (in two sessions: Colab reclaimed the first VM, and training resumed from a saved checkpoint).

## Data and licenses

- Code: MIT.
- Base model: [NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16), under the NVIDIA Open Model License; the trained adapter and merged weights inherit it.
- The judge requests come from mu's own coding sessions and from generated scenarios. They are not included: sessions hold their authors' content. `rows_from_sessions.py` builds the same kind of data from your own sessions.
