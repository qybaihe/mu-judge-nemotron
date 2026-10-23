"""Merge a trained LoRA adapter into Nemotron 3 Nano 4B, giving the folder serve/systemone_server.py loads.

    python pipeline/merge.py --adapter runs/judge/adapter --out runs/judge/merged

train.py already writes <out>/merged; this rebuilds it from the adapter alone (about 140 MB instead of 8 GB), on any
machine with the base model, a GPU or not.
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE, help="the base model the adapter was trained on")
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    # As in train.py: transformers 5 refuses to save the base generation config (top_p without do_sample).
    merged.generation_config.do_sample = True
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print("merged", args.adapter, "into", args.out, flush=True)


if __name__ == "__main__":
    main()
