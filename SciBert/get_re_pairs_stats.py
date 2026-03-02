#!/usr/bin/env python3
"""
Count RE training pair distribution across all folds.

Runs BEFORE training — no models needed.
Calls extract_pairs_from_chunk from 04b directly so the counting
logic is identical to what build_rel_dataset uses.

OUTPUT:
    outputs/re_pair_stats.csv  — one row per (fold, label)
"""

import json
import pandas as pd
from SciBert.config import Config
from SciBert.dII_train_relation import extract_pairs_from_chunk


def count_pairs_for_fold(fold: int) -> list[dict]:
    train_file = Config.OUTPUTS / f"train_chunks_fold{fold}.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Run 03 first: {train_file}")

    with open(train_file, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    label_counts: dict[str, int] = {}
    for chunk in chunks:
        for _, label in extract_pairs_from_chunk(chunk):
            label_counts[label] = label_counts.get(label, 0) + 1

    total = sum(label_counts.values())
    return [
        {"fold": fold, "label": label, "count": count, "pct": round(100 * count / total, 1)}
        for label, count in sorted(label_counts.items())
    ]


def main():
    Config.validate()

    rows = []
    for fold in range(1, Config.N_FOLDS + 1):
        rows.extend(count_pairs_for_fold(fold))

    df = pd.DataFrame(rows)
    out = Config.OUTPUTS / "re_pair_stats.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()