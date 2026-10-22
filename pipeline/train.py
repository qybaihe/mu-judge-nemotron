"""Distil the teacher's label distributions into NVIDIA Nemotron 3 Nano 4B with LoRA.

The student sees one judge question at a time (judge_format.messages), rendered with the model's chat template and
thinking off (the prompt ends in `<think></think>`), and is trained so that its first token's distribution over the
question's label tokens matches the teacher's:

  loss = -sum_l teacher(l) * log p_student(label token l | prompt)

The log-probabilities are over the whole vocabulary, so the same loss also moves probability off non-label tokens:
the student learns to answer in the format as well as what to answer.

Examples run in left-padded micro-batches of similar length; at start-up a few are checked against unpadded runs.

  python train.py --train train.jsonl --val val.jsonl --out /content/run   (on a GPU)
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# peft cannot wrap the Mamba mixers' out_proj and conv1d; lm_head stays frozen.
EXCLUDED_MODULES = {"lm_head", "out_proj", "conv1d"}


def load(path, tokenizer, max_len):
    examples, skipped = [], 0
    label_ids = {}
    with open(path, encoding="utf-8") as source:
        for line in source:
            example = json.loads(line)
            ids = tokenizer.apply_chat_template(
                example["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=True
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if len(ids) > max_len:
                skipped += 1
                continue
            targets = []
            for label in example["labels"]:
                if label not in label_ids:
                    encoded = tokenizer.encode(label, add_special_tokens=False)
                    assert len(encoded) == 1, f"label {label!r} is {len(encoded)} tokens"
                    label_ids[label] = encoded[0]
                targets.append(label_ids[label])
            examples.append(
                {"ids": ids, "label_ids": targets, "target": example["target"], "spec": example["spec"],
                 "type": example.get("type", "")}
            )
    return examples, skipped


def batch_logprobs(model, batch, pad_id, device):
    """Log-softmax of the first answer token for each example, left-padded into one forward pass."""
    width = max(len(e["ids"]) for e in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, e in enumerate(batch):
        input_ids[row, width - len(e["ids"]) :] = torch.tensor(e["ids"])
        mask[row, width - len(e["ids"]) :] = 1
    logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), logits_to_keep=1).logits[:, -1]
    return torch.log_softmax(logits.float(), dim=-1)


def loss_of(logprobs_row, example, device):
    label_logprobs = logprobs_row[torch.tensor(example["label_ids"], device=device)]
    target = torch.tensor(example["target"], device=device)
    return -(target * label_logprobs).sum(), label_logprobs


def micro_batches(examples, indices, size):
    """Indices grouped into micro-batches of similar length."""
    ordered = sorted(indices, key=lambda i: len(examples[i]["ids"]))
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


@torch.no_grad()
def evaluate(model, examples, pad_id, device, micro, limit=None):
    model.eval()
    rows = examples if limit is None else examples[:limit]
    per = defaultdict(lambda: {"n": 0, "loss": 0.0, "agree": 0, "mass": 0.0, "brier": 0.0})
    for group in micro_batches(rows, range(len(rows)), micro):
        batch = [rows[i] for i in group]
        logprobs = batch_logprobs(model, batch, pad_id, device)
        for row, example in enumerate(batch):
            loss, label_logprobs = loss_of(logprobs[row], example, device)
            probs = label_logprobs.exp()
            mass = probs.sum().item()
            normalized = probs / max(mass, 1e-9)
            target = torch.tensor(example["target"], device=device)
            for key in ("all", example["spec"], f"type:{example['type']}"):
                stats = per[key]
                stats["n"] += 1
                stats["loss"] += loss.item()
                stats["agree"] += int(normalized.argmax().item() == target.argmax().item())
                stats["mass"] += mass
                stats["brier"] += ((normalized - target) ** 2).sum().item()
    model.train()
    return {
        key: {
            "n": s["n"],
            "loss": round(s["loss"] / s["n"], 4),
            "agree": round(s["agree"] / s["n"], 4),
            "label_mass": round(s["mass"] / s["n"], 4),
            "brier": round(s["brier"] / s["n"], 4),
        }
        for key, s in per.items()
    }


@torch.no_grad()
def check_padding(model, examples, pad_id, device):
    """Padded micro-batches must give the same first-token distribution as one example at a time."""
    model.eval()
    sample = sorted(random.sample(examples, min(8, len(examples))), key=lambda e: len(e["ids"]))
    batched = batch_logprobs(model, sample, pad_id, device)
    worst = 0.0
    for row, e in enumerate(sample):
        alone = batch_logprobs(model, [e], pad_id, device)[0]
        ids = torch.tensor(e["label_ids"], device=device)
        worst = max(worst, (batched[row][ids].exp() - alone[ids].exp()).abs().max().item())
    model.train()
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accum", type=int, default=32, help="examples per optimizer step")
    parser.add_argument("--micro", type=int, default=8, help="examples per forward pass")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=150, help="optimizer steps")
    parser.add_argument("--eval-limit", type=int, default=1500)
    parser.add_argument("--baseline-only", action="store_true", help="evaluate the untuned model and stop")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)

    train, skipped_train = load(args.train, tokenizer, args.max_len)
    val, skipped_val = load(args.val, tokenizer, args.max_len)
    random.shuffle(val)
    print(f"train {len(train)} (skipped {skipped_train} too long), val {len(val)} (skipped {skipped_val})", flush=True)

    drift = check_padding(model, train, pad_id, device)
    print(f"padding check: max label-probability difference {drift:.4f}", flush=True)
    # bf16 differs by up to ~0.035 between batch shapes (left or right padding alike); a masking bug is far larger.
    micro = args.micro if drift < 0.06 else 1
    if micro != args.micro:
        print("padded batches disagree with single examples; training one example at a time", flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    baseline = evaluate(model, val, pad_id, device, micro, args.eval_limit)
    print("baseline", json.dumps(baseline["all"]), flush=True)
    log.write(json.dumps({"step": 0, "eval": baseline, "padding_check": drift}) + "\n")
    log.flush()
    if args.baseline_only:
        return

    targets = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)} - EXCLUDED_MODULES)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules=targets))
    model.print_trainable_parameters()
    model.train()

    total_steps = math.ceil(len(train) * args.epochs / args.accum)
    warmup = max(1, total_steps // 20)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))),
    )

    order = []
    while len(order) < total_steps * args.accum:
        epoch = list(range(len(train)))
        random.shuffle(epoch)
        order.extend(epoch)
    started = time.time()
    running = 0.0
    for step in range(total_steps):
        chunk = order[step * args.accum : (step + 1) * args.accum]
        for group in micro_batches(train, chunk, micro):
            batch = [train[i] for i in group]
            logprobs = batch_logprobs(model, batch, pad_id, device)
            loss = sum(loss_of(logprobs[row], e, device)[0] for row, e in enumerate(batch))
            (loss / len(chunk)).backward()
            running += loss.item() / len(chunk)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0:
            elapsed = time.time() - started
            print(
                f"step {step + 1}/{total_steps} loss {running / 10:.4f} lr {schedule.get_last_lr()[0]:.2e} "
                f"{elapsed / (step + 1):.1f}s/step eta {(total_steps - step - 1) * elapsed / (step + 1) / 60:.0f}m",
                flush=True,
            )
            log.write(json.dumps({"step": step + 1, "train_loss": running / 10}) + "\n")
            running = 0.0
        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            result = evaluate(model, val, pad_id, device, micro, args.eval_limit)
            print("eval", step + 1, json.dumps(result["all"]), flush=True)
            log.write(json.dumps({"step": step + 1, "eval": result}) + "\n")
            log.flush()
            model.save_pretrained(os.path.join(args.out, "adapter"))

    final = evaluate(model, val, pad_id, device, micro)
    print("final", json.dumps(final["all"]), flush=True)
    log.write(json.dumps({"step": total_steps, "final": final}) + "\n")
    log.close()
    model.save_pretrained(os.path.join(args.out, "adapter"))
    merged = model.merge_and_unload()
    # The base model's generation config sets top_p without do_sample, which transformers 5 refuses to save.
    # Sampling is what NVIDIA recommends for this model; the judge itself reads first-token probabilities.
    merged.generation_config.do_sample = True
    merged.save_pretrained(os.path.join(args.out, "merged"), safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(args.out, "merged"))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
