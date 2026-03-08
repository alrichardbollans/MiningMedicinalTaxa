
"""
Train NER models for all 5 folds.
We train using AdamW with a linear learning rate decay scheduler type

USAGE:
    python SciBert/04a_train_ner.py              # Train all 5 folds
    python SciBert/04a_train_ner.py --fold 1     # Train single fold

OUTPUT:
    models/ner_scibert_lora_fold{1-5}/          # save the final model for each iteration of the cv
    outputs/ner_training_log_fold{1-5}.csv      # log for loss
    outputs/ner_training_summary.csv            # final summary

"""

import sys
import json
import shutil
import argparse
from typing import List, Tuple

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from SciBert.config import Config


def assign_bio_label(token_start: int, token_end: int, entity_spans: List[Tuple[int, int, str]]) -> str:
    """
    It assigns label BIO to a token given a span of characters in the text.
    Spans come from entities in the JSONL obtained with 01_chunk_data.py.
    Offset of tokens come from tokenizer with return_offsets_mapping=True.

    """
    for entity_start, entity_end, entity_type in entity_spans:
        if token_start >= entity_start and token_end <= entity_end:
            if token_start == entity_start:
                return f"B-{entity_type}"
            else:
                return f"I-{entity_type}"
    return "O"


def build_ner_dataset(chunks: list, tokenizer, max_length: int) -> Dataset:
    """
    Converts chunk JSONL in datasets.Dataset that works with Trainer HuggingFace see
    https://stackoverflow.com/questions/70127516/how-to-create-a-torch-utils-data-dataset-and-import-it-into-a-torch-utils-data-d
    Padding is fixed at a max_length.
    -100 for special tokens ([CLS], [SEP], [PAD]) ignored from CrossEntropy Loss
    Ref: https://huggingface.co/docs/transformers/tasks/token_classification
    """
    records = []

    for chunk in chunks:
        text = chunk['text']
        entity_spans = sorted(
            [(e['start'], e['end'], e['label']) for e in chunk.get('entities', [])],
            key=lambda x: (x[0], x[1])
        )

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )

        labels = []
        for start_char, end_char in encoding['offset_mapping']:
            if start_char == end_char:
                # Special tokens ignored by loss ([CLS], [SEP], [PAD])
                labels.append(-100)
            else:
                labels.append(Config.LABEL2ID[assign_bio_label(start_char, end_char, entity_spans)])

        records.append({
            'input_ids':      encoding['input_ids'],       # lista di int
            'attention_mask': encoding['attention_mask'],  # lista di int
            'labels':         labels,                      # lista di int
        })

    return Dataset.from_list(records)


def train_fold(fold: int, tag: str = None) -> dict:
    """Function for single NER for one fold it outputs dict for csv summary"""
    cfg = Config.NER_RE_FULL
    tag = tag or f"fold{fold}"

    train_file = Config.OUTPUTS / f"train_chunks_{tag}.jsonl"
    model_dir  = Config.MODELS  / f"ner_scibert_lora_{tag}"
    temp_dir   = Config.MODELS  / f"ner_{tag}_temp"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    with open(train_file, encoding='utf-8') as f:
        train_chunks = [json.loads(line) for line in f]

    train_dataset = build_ner_dataset(train_chunks, tokenizer, cfg["max_length"])

    base_model = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.BIO_LABELS),
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config)
    # model.print_trainable_parameters()
    #trainable params: 591,362 || all params: 109,920,772 || trainable%: 0.5380
    # out of 110 million parameters we are training only 0.54%
    # https://huggingface.co/docs/peft

    steps_per_epoch = len(train_dataset) // cfg["batch_size"]
    total_steps     = steps_per_epoch * cfg["num_epochs"]
    warmup_steps    = int(total_steps * cfg["warmup_ratio"])

    temp_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(temp_dir),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        warmup_steps=warmup_steps,
        weight_decay=cfg["weight_decay"],
        per_device_train_batch_size=cfg["batch_size"],
        logging_steps=max(10, steps_per_epoch // 10),
        report_to=[],
        fp16=True, # see https://medium.com/@staytechrich/multi-gpu-training-with-hugging-face-transformers-a-complete-guide-ab2cf241df94
        dataloader_num_workers= 2,
        disable_tqdm=False, # see realtime progress while training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Complete log including loss, learning_rate, epoch, step
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.insert(0, 'fold', tag)
    log_df.to_csv(Config.OUTPUTS / f"ner_training_log_fold{fold}.csv", index=False)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    train_rows = [r for r in trainer.state.log_history if 'loss' in r]

    return {
        'fold':             tag,
        'train_chunks':     len(train_chunks),
        'final_train_loss': train_rows[-1]['loss'] if train_rows else None,
        'status': 'success',
        'error': None,
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
                'fold': fold,
                'train_chunks': None,
                'final_train_loss': None,
                'status': 'failed',
                'error': str(e),
            }
        rows.append(row)

    summary_path = Config.OUTPUTS / "ner_training_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Train NER models')
    parser.add_argument('--fold', type=int, help='Train single fold (1-5)')
    args = parser.parse_args()

    Config.validate()

    if args.fold:
        if args.fold < 1 or args.fold > Config.N_FOLDS:
            print(f"Error: fold must be between 1 and {Config.N_FOLDS}")
            sys.exit(1)
        row = train_fold(args.fold)
        pd.DataFrame([row]).to_csv(
            Config.OUTPUTS / f"ner_training_summary_fold{args.fold}.csv", index=False
        )
    else:
        train_all_folds()


if __name__ == "__main__":
    main()