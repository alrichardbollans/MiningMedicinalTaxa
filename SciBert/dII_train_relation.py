#!/usr/bin/env python3
# 04b_train_relation.py
"""
Train Relation Extraction models for all 5 folds.

USAGE:
    python 04b_train_relation.py              # Train all 5 folds
    python 04b_train_relation.py --fold 1     # Train single fold

OUTPUT:
    models/re_scibert_lora_fold{1-5}/         # saved model per fold
    outputs/re_training_log_fold{1-5}.csv     # per-step loss log
    outputs/re_training_summary.csv           # final summary across folds
    outputs/re_dataset_stats_fold{1-5}.csv    # relation distribution per fold
"""

import sys
import json
import shutil
import random
import argparse
from typing import List, Tuple, Dict

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from SciBert.config import Config





# ---Text marking -----

def create_marked_text(text: str, entity1: dict, entity2: dict) -> str:
    """
    Insert entity markers around the two entities in the text.

    [E1]/[/E1] always marks entity1 (head = Scientific Name).
    [E2]/[/E2] always marks entity2 (tail = Effect or Condition).

    Insertion is done in reverse textual order to avoid offset shifts.

    NOTE: entity1['start'] < entity2['start'] is the common case in biomedical
    text (plant name before effect), but we handle the reverse without swapping
    the semantic role of the markers.
    """
    # Build list of insertions: (position, text_to_insert)
    # Possible experiment with entity typed markers e.g. Sci Names, Medical Condition etc as shown in https://aclanthology.org/2022.aacl-short.21.pdf
    insertions = [
        (entity1['start'], '[E1]'),
        (entity1['end'],   '[/E1]'),
        (entity2['start'], '[E2]'),
        (entity2['end'],   '[/E2]'),
    ]
    # Apply in reverse order so earlier offsets stay valid
    insertions.sort(key=lambda x: x[0], reverse=True)

    result = text
    for pos, marker in insertions:
        result = result[:pos] + marker + result[pos:]

    # Normalize whitespace
    return ' '.join(result.split())

# ---Pair extraction---


def extract_pairs_from_chunk(chunk: dict) -> List[Tuple[str, str]]:
    """
    Extract (marked_text, relation_label) pairs from one chunk.

    relation_map uses (i, j) positional indices into the entities array.
    This matches how relations are stored in the JSONL (head/tail are indices).

    All valid pairs are included (no negative sampling)
    """
    text = chunk['text']
    entities = chunk.get('entities', [])
    relations = chunk.get('relations', [])

    relation_map: Dict[Tuple[int, int], str] = {}
    for rel in relations:
        relation_map[(rel['head'], rel['tail'])] = rel['label']

    pairs = []

    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i == j:
                continue

            head_type = entity1.get('label', entity1.get('type', ''))
            tail_type = entity2.get('label', entity2.get('type', ''))
            allowed = Config.valid_relation_types(head_type, tail_type)

            if not allowed:
                continue

            marked_text = create_marked_text(text, entity1, entity2)

            if (i, j) in relation_map:
                label = relation_map[(i, j)]
            else:
                label = 'no_relation'

            pairs.append((marked_text, label))

    return pairs



#--- Dataset construction---

def build_rel_dataset(chunks: list, tokenizer, max_length: int, negative_sample_ratio = None) -> Tuple[Dataset, List[Tuple[str, str]]]: # this outputs all_pairs
    """
    Build a datasets.Dataset of tokenized relation pairs from a list of chunks.

    Tokenization is done upfront (same pattern as build_ner_dataset in 04a).

    """
    all_pairs: List[Tuple[str, str]] = []
    for chunk in chunks:
        all_pairs.extend(extract_pairs_from_chunk(chunk))
    # NEGATIVE SAMPLE
    if negative_sample_ratio is not None:
        positives = [(t, l) for t, l in all_pairs if l != 'no_relation']
        negatives = [(t, l) for t, l in all_pairs if l == 'no_relation']
        n_keep = min(len(negatives), negative_sample_ratio * len(positives))
        negatives = random.sample(negatives, n_keep)
        all_pairs = positives + negatives
        random.shuffle(all_pairs)


    records = []
    for marked_text, label in all_pairs:
        encoding = tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
        encoding['labels'] = Config.REL_LABEL2ID[label]
        records.append({
            'input_ids':      encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels':         encoding['labels'],
        })

    return Dataset.from_list(records), all_pairs



# ----Training----

def train_fold(fold: int, tag: str = None) -> dict:
    """Train RE model for one fold. Returns a dict row for the summary CSV."""
    cfg = Config.NER_RE_FULL
    tag = tag or f"fold{fold}"

    train_file = Config.OUTPUTS / f"train_chunks_{tag}.jsonl"
    model_dir  = Config.MODELS  / f"re_scibert_lora_{tag}"
    temp_dir   = Config.MODELS  / f"re_{tag}_temp"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Add entity marker tokens: required before resize_token_embeddings
    special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    with open(train_file, encoding='utf-8') as f:
        train_chunks = [json.loads(line) for line in f]

    random.seed(42)
    train_dataset, all_pairs = build_rel_dataset(train_chunks, tokenizer, cfg['max_length'], cfg.get('negative_sample_ratio')) # add neg sample ratio

    label_counts = {}
    for _, label in all_pairs:
        label_counts[label] = label_counts.get(label, 0) + 1
    total = len(all_pairs)
    stats = [{'fold': fold, 'label': l, 'count': c, 'pct': round(100 * c / total, 1)}
             for l, c in sorted(label_counts.items())]
    pd.DataFrame(stats).to_csv(Config.OUTPUTS / f"re_dataset_stats_fold{fold}.csv", index=False)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.RELATION_TYPES),
        id2label=Config.REL_ID2LABEL,
        label2id=Config.REL_LABEL2ID,
    )

    # Resize embeddings to cover the 4 new special tokens
    # Without this the new token ids have no embedding rows in the model.
    # Ref: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    base_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        target_modules=['query', 'value'],
    )
    model = get_peft_model(base_model, lora_config)

    steps_per_epoch = len(train_dataset) // cfg['batch_size']
    total_steps     = steps_per_epoch * cfg['num_epochs']
    warmup_steps    = int(total_steps * cfg['warmup_ratio'])

    temp_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(temp_dir),
        learning_rate=cfg['learning_rate'],
        num_train_epochs=cfg['num_epochs'],
        warmup_steps=warmup_steps,
        weight_decay=cfg['weight_decay'],
        per_device_train_batch_size=cfg['batch_size'],
        logging_steps=max(10, steps_per_epoch // 10),
        report_to=[],
        fp16=True,
        dataloader_num_workers=2,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset intentionally omitted — test fold must not be seen during training
    )

    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Per-step loss log
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.insert(0, 'fold', tag)
    log_df.to_csv(Config.OUTPUTS / f"re_training_log_fold{fold}.csv", index=False)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    train_rows = [r for r in trainer.state.log_history if 'loss' in r]

    return {
        'fold':             tag,
        'train_chunks':     len(train_chunks),
        'train_pairs':      len(train_dataset),
        'final_train_loss': train_rows[-1]['loss'] if train_rows else None,
        'status':           'success',
        'error':            None,
    }


def train_all_folds():
    rows = []
    for fold in range(1, Config.N_FOLDS + 1):
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
        try:
            row = train_fold(fold)
        except Exception as e:
            print(f"ERROR fold {fold}: {e}")
            row = {
                'fold':             fold,
                'train_chunks':     None,
                'train_pairs':      None,
                'final_train_loss': None,
                'status':           'failed',
                'error':            str(e),
            }
        rows.append(row)

    summary_path = Config.OUTPUTS / "re_training_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Relation Extraction models')
    parser.add_argument('--fold', type=int, help='Train single fold (1-5)')
    args = parser.parse_args()

    Config.validate()

    if args.fold:
        if args.fold < 1 or args.fold > Config.N_FOLDS:
            print(f"Error: fold must be between 1 and {Config.N_FOLDS}")
            sys.exit(1)
        row = train_fold(args.fold)
        pd.DataFrame([row]).to_csv(
            Config.OUTPUTS / f"re_training_summary_fold{args.fold}.csv", index=False
        )
    else:
        train_all_folds()


if __name__ == "__main__":
    main()