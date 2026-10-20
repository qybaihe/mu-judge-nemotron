Data is not committed. `pipeline/` rebuilds it:

- `raw/`: downloads (Nebius trajectory parquet files, tokenizer files).
- `transcripts/`: `convert_openhands.py` output.
- `cases/`: mu's `kyrn/distill/export-cases.ts` output.
- `labels/`: `label.py` output.
- `dataset/`: `build_dataset.py` output.
