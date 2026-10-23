---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
library_name: transformers
tags:
- nemotron
- judge
- coding-agent
- lora
---

# mu-judge-nemotron (Nemotron 3 Nano 4B, fine-tuned as mu's judge)

The judge model of the [mu](https://github.com/qybaihe/mu) coding agent, for running on your own machine. mu asks a judge bounded questions at 40 decision points in its agent loop (is this command safe for this task, is this test output worth reading, did the agent stop short, has it drifted). This model answers each question with one token, so every answer has a probability: `yes`/`no`, `A`–`I` for options, `0`–`3` for rubric levels.

- **Base:** NVIDIA Nemotron 3 Nano 4B (BF16), LoRA r=32 on the attention and MLP projections, merged.
- **Data:** 9,494 judge requests (about 31,500 questions) from mu's sessions and generated scenarios, each question with a probability for every answer label. No request in the evaluation sets below, and no coding session one comes from, is among them.
- **Objective:** soft cross-entropy between each question's label distribution and the model's first-token distribution.
- **Training:** one epoch (939 optimizer steps), about 3.1 hours on one A100 40 GB; about 30,000 training examples (one per question), 928 held out for validation. On the validation questions its top answer agrees with the label's top answer 88.8% of the time (Brier score 0.106, from 0.304 untuned).

## Use

Serve it over System One with [mu-judge-nemotron](https://github.com/qybaihe/mu-judge-nemotron)'s server, then use mu's built-in `nemotron-local` judge:

```bash
python serve/systemone_server.py --model qybaihe/mu-judge-nemotron-nano-4b --port 8770
```

```json
{ "tiers": ["nemotron-local"] }
```

The prompt format is fixed: see `pipeline/judge_format.py`. A question asked any other way is out of distribution.

## Evaluation

1,353 held-out judge requests (541 real, from coding sessions; 700 constructed scenarios; 112 written by mu's author). No request in this set, and no coding session one comes from, was used in training. Score: the share of the cost of mu's no-judge fallbacks that the judge removes, judged by what mu then does (0 = no better than no judge).

| Judge | Real requests | Constructed | Author-written |
| --- | --- | --- | --- |
| **This model** | **62.3** | 45.0 | 19.6 |
| This model, then Ultra for the questions it is unsure of | 63.1 | 44.1 | 38.6 |
| Nemotron 3 Ultra, thinking off | 47.3 | 47.7 | 52.3 |
| Nemotron 3 Nano 4B, untuned | −3.6 | 13.0 | −11.8 |

On 4,838 further held-out requests, scored once (95% bootstrap intervals over coding tasks and scenarios):

| Judge | Real requests | Constructed |
| --- | --- | --- |
| **This model** | **68.2** (61.0 to 76.0) | **59.9** (51.5 to 68.3) |
| Nemotron 3 Ultra, thinking off | 54.7 (44.6 to 64.0) | 48.4 (35.8 to 59.8) |
| This model, then Ultra for the questions it is unsure of | 20.0 (−74.6 to 69.7) | 55.7 (44.8 to 66.4) |

Paired against Ultra it is ahead by 13.6 on real requests (+8.1 to +20.7) and by 11.5 on constructed ones (+1.5 to +22.6). Sending its unsure questions to Ultra did not pay on real test requests: in a few costly places Ultra's confident answers cost more than this model's doubt.

## Limitations

- English and Chinese coding-agent states only; other languages are untested.
- It is clearly weaker than Ultra on the hard, deliberately tricky author-written requests (19.6 against 52.3): a cascade with Ultra behind it (`"tiers": ["nemotron-local", "nemotron"]`) recovered part of that on dev, but did worse than this model alone on real test requests.
- Calibrated for mu's thresholds (no ≤ 0.2, yes ≥ 0.8). For any other use, read its probabilities with care.

## License

The weights are a derivative of NVIDIA Nemotron 3 Nano 4B and are released under the NVIDIA Open Model License. The training and serving code is MIT.
