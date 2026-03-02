#!/usr/bin/env python3
# 03_create_chunks.py
"""
Assign subchunks into train/test sets based on cunk-level (or task level) fold assignments.

- Subchunks were created from ALL chunks (taskl) in the json annotated file, using 1_chunk_data_essential
- Now we need to assign them to cross-validation folds
- Is important to ensures task-level separation (no leakage)

PROCESS:
1. Load fold assignments (which task IDs in train vs test)
2. Read all subchunks from scibert_chunks.jsonl
3. Route each chunk to train or test based on its task_id
4. Write separate JSONL files for each fold

OUTPUT:
- outputs/train_chunks_fold{1-5}.jsonl
- outputs/test_chunks_fold{1-5}.jsonl
"""

import json
import pandas as pd
from SciBert.config import Config


def split_fold(fold_num, chunks_file, all_valid_ids):
    with open(Config.SPLITS / f"train_ids_fold{fold_num}.json") as f:
        train_ids = set(json.load(f))
    with open(Config.SPLITS / f"test_ids_fold{fold_num}.json") as f:
        test_ids = set(json.load(f))

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Fold {fold_num} data leakage: {overlap}")

    train_chunks, test_chunks, orphan_ids = [], [], set()

    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            task_id = chunk["task_id"]
            if task_id in train_ids:
                train_chunks.append(chunk)
            elif task_id in test_ids:
                test_chunks.append(chunk)
            elif task_id in all_valid_ids:
                orphan_ids.add(task_id)

    if orphan_ids:
        print(f"WARNING fold {fold_num}: orphan doc IDs {sorted(orphan_ids)} — bug in 02_create_splits.py")

    for chunks, name in [(train_chunks, "train"), (test_chunks, "test")]:
        with open(Config.OUTPUTS / f"{name}_chunks_fold{fold_num}.jsonl", "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return len(train_chunks), len(test_chunks)


def main():
    Config.validate()

    chunks_file = Config.OUTPUTS / "scibert_chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Run 01_chunk_data.py first: {chunks_file}")

    all_valid_ids = (
            set(pd.read_csv(Config.TUNING_CSV)["id"]) |
            set(pd.read_csv(Config.TESTING_CSV)["id"])
    )

    summary = []
    for fold_num in range(1, Config.N_FOLDS + 1):
        train_n, test_n = split_fold(fold_num, chunks_file, all_valid_ids)
        summary.append({"fold": fold_num, "train_chunks": train_n, "test_chunks": test_n})
        print(f"Fold {fold_num}: {train_n} train, {test_n} test")

    pd.DataFrame(summary).to_csv(Config.OUTPUTS / "split_chunks_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()