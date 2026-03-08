#!/usr/bin/env python3
# 05_end_to_end_prediction.py
"""
End-to-end inference NER + Relation Extraction TO TaxaData output.

    dI trains NER models, dII trains RE models.
    In this script, for each fold loads the trained models, it runs them on the held-out test chunks, and saves
    predictions.

PREREQUISITES:
    - dI_train_ner.py completed (models/ner_scibert_lora_fold{1-5}/)
    - dII_train_relation.py completed (models/re_scibert_lora_fold{1-5}/)
OUTPUT:
    outputs/predictions_fold{1-5}_raw.json      - entity spans + relation indices (check and debug)
    outputs/predictions_fold{1-5}_taxadata.json - TaxaData format for eval

USAGE:
    python V_end_to_end_prediction.py --fold 1
    python V_end_to_end_prediction.py          # all folds
"""

import sys
import json
import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from peft import PeftModel

from SciBert.config import Config
from SciBert.dII_train_relation import create_marked_text
from LLM_models.structured_output_schema import Taxon, TaxaData


#---Model loading----


def load_models_from_path(ner_path: str, re_path: str) -> dict:
    """
    Load NER and RE models from a local path or HuggingFace repo ID.

    Accepts either a local directory path or a HF repo ID (e.g. 'Ficco84/ner-scibert-medicinal-plants').
    For private HF repos, call huggingface_hub.login() before calling this function.

    Returns a dict with keys: ner_model, ner_tokenizer, re_model, re_tokenizer, device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ner_tokenizer = AutoTokenizer.from_pretrained(ner_path)
    ner_base = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.BIO_LABELS),
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID,
    )
    ner_model = PeftModel.from_pretrained(ner_base, ner_path)
    ner_model.eval()
    ner_model = ner_model.to(device)

    re_tokenizer = AutoTokenizer.from_pretrained(re_path)
    re_base = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.RELATION_TYPES),
        id2label=Config.REL_ID2LABEL,
        label2id=Config.REL_LABEL2ID,
    )
    re_base.resize_token_embeddings(len(re_tokenizer))
    re_model = PeftModel.from_pretrained(re_base, re_path)
    re_model.eval()
    re_model = re_model.to(device)

    return {
        'ner_model':     ner_model,
        'ner_tokenizer': ner_tokenizer,
        're_model':      re_model,
        're_tokenizer':  re_tokenizer,
        'device':        device,
    }


def load_models(fold) -> dict:
    """Load NER and RE models for a given fold from local path."""
    return load_models_from_path(
        str(Config.MODELS / f"ner_scibert_lora_fold{fold}"),
        str(Config.MODELS / f"re_scibert_lora_fold{fold}"),
    )

#--- NER inference ----
def predict_entities(text: str, models: dict) -> List[Dict]:
    """
    Run NER model on text, return list of entity dicts with keys:
    label, start, end, text.

    BIO tags are collapsed into spans via offset mapping.
    Subword tokens belonging to the same entity are merged by
    extending the span end and tracking continuity.
    """
    ner_tokenizer = models['ner_tokenizer']
    ner_model     = models['ner_model']
    device        = models['device']

    enc = ner_tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    offsets = enc.pop('offset_mapping')[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = ner_model(**enc).logits[0]

    pred_ids = logits.argmax(-1).cpu().tolist()
    tokens   = ner_tokenizer.convert_ids_to_tokens(enc['input_ids'][0].cpu())

    entities = []
    current  = None

    for tok, label_id, (start, end) in zip(tokens, pred_ids, offsets):
        if tok in ['[CLS]', '[SEP]', '[PAD]'] or start == end:
            if current:
                entities.append(current)
                current = None
            continue
      # Force-merge subword tokens into current entity
        if tok.startswith('##'):
            if current is not None:
                current['end'] = max(current['end'], end)
            continue
        label = Config.ID2LABEL[label_id]

        if label == 'O':
            if current:
                entities.append(current)
                current = None
            continue

        entity_type = label.split('-', 1)[1] if '-' in label else label

        if current is None:
            current = {'label': entity_type, 'start': start, 'end': end}
        elif current['label'] == entity_type and start <= current['end'] + 1:
            current['end'] = max(current['end'], end)
        else:
            entities.append(current)
            current = {'label': entity_type, 'start': start, 'end': end}

    if current:
        entities.append(current)

    for ent in entities:
        ent['text'] = text[ent['start']:ent['end']]

    return entities


# --- RE inference ---

def predict_relations(text: str, entities: List[Dict], models: dict) -> List[Dict]:
    """
    Run RE model on all valid entity pairs.

    Valid pairs are determined by Config.valid_relation_types which is the same constraint
    used during training in IVb. The function create_marked_text is imported from IVb.

    Only treats_medical_condition and has medicinal effect predictions are returned. The no relation is ignored
    """
    re_tokenizer = models['re_tokenizer']
    re_model     = models['re_model']
    device       = models['device']

    relations = []

    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i == j:
                continue

            if not Config.valid_relation_types(entity1.get('label', ''), entity2.get('label', '')):
                continue

            marked_text = create_marked_text(text, entity1, entity2)

            enc = re_tokenizer(
                marked_text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = re_model(**enc).logits[0]

            pred_label = Config.REL_ID2LABEL[logits.argmax().item()]

            if pred_label != 'no_relation':
                relations.append({'head': i, 'tail': j, 'label': pred_label})

    return relations


#---TaxaData conversion ---


def convert_to_taxadata(entities: List[Dict], relations: List[Dict]) -> TaxaData:
    """
    Convert raw NER + RE predictions to TaxaData format.

    Scientific names (plant or fungus) become Taxon entries.
    Relations map effects and conditions onto the corresponding taxon.
    Scientific names with no relations are kept as lone taxa
    (medicinal_effects=None, medical_conditions=None) — consistent with
    how ground truth is built in 08a/08b.
    """
    scientific_name_labels = {'Scientific Plant Name', 'Scientific Fungus Name'}
    taxa: Dict[str, Taxon] = {}

    for entity in entities:
        if entity['label'] in scientific_name_labels:
            sci_name = entity['text'].lower()
            if sci_name not in taxa:
                taxa[sci_name] = Taxon(
                    scientific_name=sci_name,
                    medical_conditions=None,
                    medicinal_effects=None,
                )

    for rel in relations:
        head_entity = entities[rel['head']]
        tail_entity = entities[rel['tail']]

        if head_entity['label'] not in scientific_name_labels:
            continue

        sci_name  = head_entity['text'].lower()
        tail_text = tail_entity['text'].lower()

        if sci_name not in taxa:
            continue

        taxon = taxa[sci_name]

        if rel['label'] == 'has_medicinal_effect':
            if taxon.medicinal_effects is None:
                taxon.medicinal_effects = []
            if tail_text not in taxon.medicinal_effects:
                taxon.medicinal_effects.append(tail_text)

        elif rel['label'] == 'treats_medical_condition':
            if taxon.medical_conditions is None:
                taxon.medical_conditions = []
            if tail_text not in taxon.medical_conditions:
                taxon.medical_conditions.append(tail_text)

    return TaxaData(taxa=list(taxa.values()))



# ---Fold prediction loop ---
def predict_fold(fold: int) -> None:
    """Load models for one fold, run inference on all test chunks, save outputs."""
    test_file = Config.OUTPUTS / f"test_chunks_fold{fold}.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

    models = load_models(fold)

    with open(test_file, encoding='utf-8') as f:
        test_chunks = [json.loads(line) for line in f]

    print(f"Predicting {len(test_chunks)} chunks...")

    all_raw      = []
    all_taxadata = []

    for i, chunk in enumerate(test_chunks):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_chunks)}")

        text      = chunk['text']
        entities  = predict_entities(text, models)
        relations = predict_relations(text, entities, models)
        taxadata  = convert_to_taxadata(entities, relations)

        all_raw.append({
            'chunk_id': chunk.get('id', i),
            'task_id':  chunk.get('task_id'),
            'text':     text,
            'entities': entities,
            'relations': relations,
        })

        all_taxadata.append({
            'chunk_id': chunk.get('id', i),
            'task_id':  chunk.get('task_id'),
            'taxadata': taxadata.model_dump(),
        })

    raw_file      = Config.OUTPUTS / f"predictions_fold{fold}_raw.json"
    taxadata_file = Config.OUTPUTS / f"predictions_fold{fold}_taxadata.json"

    with open(raw_file, 'w') as f:
        json.dump(all_raw, f, indent=2)
    with open(taxadata_file, 'w') as f:
        json.dump(all_taxadata, f, indent=2)

    print(f"Saved: {raw_file.name}, {taxadata_file.name}")

    torch.cuda.empty_cache()

# --- Run code---


def main():
    parser = argparse.ArgumentParser(description='End-to-end NER + RE inference')
    parser.add_argument('--fold', type=int, help='Single fold (1-5); omit for all folds')
    args = parser.parse_args()

    Config.validate()
    Config.print_summary()

    if args.fold and (args.fold < 1 or args.fold > Config.N_FOLDS):
        print(f"Error: fold must be between 1 and {Config.N_FOLDS}")
        sys.exit(1)

    folds = [args.fold] if args.fold else range(1, Config.N_FOLDS + 1)

    for fold in folds:
        try:
            predict_fold(fold)
        except Exception as e:
            print(f"ERROR fold {fold}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
