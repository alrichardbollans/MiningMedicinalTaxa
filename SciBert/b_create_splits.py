#!/usr/bin/env python3
"""
Create 5-fold cross-validation splits at chunk (task ID) level.
Tuning set is always in training. Testing set is split into N folds.
"""

import random
import json
import pandas as pd
from SciBert.config import Config


def create_folds(tuning_ids, testing_ids, n_folds, seed=42):
    overlap = set(tuning_ids) & set(testing_ids)
    if overlap:
        raise ValueError(f"Data leakage: {overlap} in both tuning and testing")

    random.seed(seed)
    shuffled = testing_ids.copy()
    random.shuffle(shuffled)

    # Split testing into n_folds
    fold_size = len(shuffled) // n_folds # 39
    remainder = len(shuffled) % n_folds # 4
    test_folds = []
    start = 0
    for i in range(n_folds):
            extra = 1 if i < remainder else 0 # this adds 1 extra chunk to the first 4 folds
            end = start + fold_size + extra
            test_folds.append(shuffled[start:end])
            start = end # restart from the last chunk of the i fold


    train_folds = [
        tuning_ids + [d for d in testing_ids if d not in set(test_folds[k])]
        for k in range(n_folds)
    ]

    return train_folds, test_folds


def save_folds(train_folds, test_folds, output_dir, n_folds, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_folds.json", "w") as f:
        json.dump(train_folds, f, indent=2)
    with open(output_dir / "test_folds.json", "w") as f:
        json.dump(test_folds, f, indent=2)

    for i in range(n_folds):
        with open(output_dir / f"train_ids_fold{i+1}.json", "w") as f:
            json.dump(train_folds[i], f, indent=2)
        with open(output_dir / f"test_ids_fold{i+1}.json", "w") as f:
            json.dump(test_folds[i], f, indent=2)

    summary = pd.DataFrame([
        {"fold": i + 1, "train_size": len(train_folds[i]), "test_size": len(test_folds[i])}
        for i in range(n_folds)
    ])

    summary.to_csv(output_dir / "split_summary.csv", index=False)



def main():
    Config.validate()
    tuning_ids = pd.read_csv(Config.TUNING_CSV)["id"].tolist()
    testing_ids = pd.read_csv(Config.TESTING_CSV)["id"].tolist()
    train_folds, test_folds = create_folds(tuning_ids, testing_ids, Config.N_FOLDS)

    save_folds(train_folds, test_folds, Config.SPLITS, Config.N_FOLDS, seed=42)


if __name__ == "__main__":
    main()