"""
SciBERT inference — drop-in equivalent of query_a_model.

USAGE:
    from SciBert.running_scibert import load_scibert, query_scibert

    models = load_scibert()

    output = query_scibert(models, 'mytext.txt',
                           json_dump='mytext_scibert.json',
                           json_dump_filtered='mytext_scibert_filtered.json')
"""

import json
import os

import torch
from nltk.tokenize.punkt import PunktSentenceTokenizer
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

from SciBert.a_chunk_data import build_sentence_windows
from SciBert.config import Config
from SciBert.e_prediction import predict_entities, predict_relations, convert_to_taxadata
from LLM_models.structured_output_schema import TaxaData, deduplicate_and_standardise_output_taxa_lists
from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge

from SciBert.config import Config

NER_MODEL_ID = Config.ROOT / 'models' / 'ner_scibert_lora_full'
RE_MODEL_ID  = Config.ROOT / 'models' / 're_scibert_lora_full'


def load_scibert() -> dict:
    """
    Load NER and RE models from HuggingFace.
    Call once and reuse the returned dict across multiple query_scibert calls.
    For private repos, call huggingface_hub.login() before this.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_ID)
    ner_base = AutoModelForTokenClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.BIO_LABELS),
        id2label=Config.ID2LABEL,
        label2id=Config.LABEL2ID,
    )
    ner_model = PeftModel.from_pretrained(ner_base, NER_MODEL_ID)
    ner_model.eval()
    ner_model = ner_model.to(device)

    re_tokenizer = AutoTokenizer.from_pretrained(RE_MODEL_ID)
    re_base = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.RELATION_TYPES),
        id2label=Config.REL_ID2LABEL,
        label2id=Config.REL_LABEL2ID,
    )
    re_base.resize_token_embeddings(len(re_tokenizer))
    re_model = PeftModel.from_pretrained(re_base, RE_MODEL_ID)
    re_model.eval()
    re_model = re_model.to(device)

    sent_tokenizer    = PunktSentenceTokenizer()
    scibert_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_ID)

    return {
        'ner_model':          ner_model,
        'ner_tokenizer':      ner_tokenizer,
        're_model':           re_model,
        're_tokenizer':       re_tokenizer,
        'device':             device,
        'sent_tokenizer':     sent_tokenizer,
        'scibert_tokenizer':  scibert_tokenizer,
    }


def query_scibert(models: dict, txt_path: str,
                  json_dump: str = None,
                  run_re: bool = True) -> TaxaData:
    """
    Run SciBERT NER + RE on a text file.
    - json_dump: saves raw output (all extractions)
    Returns TaxaData.
    """
    with open(txt_path, encoding='utf-8') as f:
        text = f.read()

    windows = build_sentence_windows(text, models['sent_tokenizer'], models['scibert_tokenizer'])

    all_taxa = []
    for start, end in windows:
        window_text = text[start:end]
        entities = predict_entities(window_text, models)
        relations = predict_relations(window_text, entities, models) if run_re else []
        taxadata = convert_to_taxadata(entities, relations)
        all_taxa.extend(taxadata.taxa)

    output = deduplicate_and_standardise_output_taxa_lists(all_taxa)

    if json_dump:
        with open(json_dump, 'w') as f:
            json.dump(output.model_dump(mode='json'), f)

    return output