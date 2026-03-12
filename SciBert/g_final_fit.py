#!/usr/bin/env python3
# g_train_full.py
"""
Train NER and RE on the full annotated dataset (tuning + testing IDs).
Writes train_chunks_full.jsonl then calls the existing train_fold with tag "full".

PREREQUISITES:
    - a_chunk_data.py completed  ->  outputs/scibert_chunks.jsonl

OUTPUT:
    models/ner_scibert_lora_foldfull/
    models/re_scibert_lora_foldfull/
    outputs/ner_training_log_foldfull.csv
    outputs/re_training_log_foldfull.csv

USAGE:
    python SciBert/g_train_full.py
"""

import json
import pandas as pd

from SciBert.config import Config
from SciBert.dI_train_ner import train_fold as train_ner_fold
from SciBert.dII_train_relation import train_fold as train_re_fold


def main():
    Config.validate()

    all_valid_ids = (
        set(pd.read_csv(Config.TUNING_CSV)["id"]) |
        set(pd.read_csv(Config.TESTING_CSV)["id"])
    )

    chunks_file = Config.OUTPUTS / "scibert_chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Run a_chunk_data.py first: {chunks_file}")

    out_file = Config.OUTPUTS / "train_chunks_full.jsonl"
    with open(chunks_file, encoding="utf-8") as f_in, open(out_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            chunk = json.loads(line)
            if chunk["task_id"] in all_valid_ids:
                f_out.write(line)

    train_ner_fold(fold=None, tag="full")
    train_re_fold(fold=None, tag="full")


if __name__ == "__main__":
    main()