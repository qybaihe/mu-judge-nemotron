Data is not committed. `pipeline/` rebuilds it:

- `requests/`: judge requests, `{"key", "decision", "state", "questions"}` (`rows_from_sessions.py`, `build_requests.py`).
- `labels/`: the teacher's answers to them (`label_tf.py`).
- `dataset/`: one example per question (`build_dataset.py`).
- `spend.jsonl`: every paid Token Factory call, with its tokens and price (`spend.py`).
