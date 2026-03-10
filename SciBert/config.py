# config.py
"""
Central configuration for NER + RE pipeline with SciBert

ENVIRONMENT DETECTION:
- Colab: Uses Google Drive mounted at /content/drive/MyDrive/
- Local: Uses ./SciBert/ in med plant mining

"""

import os
from pathlib import Path
import json

class Config:
    # ---- env detection ----
    @staticmethod
    def detect_environment():
        """Detect if running in Colab or local"""
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            return 'colab'
        return 'local'

    ENV = detect_environment.__func__()

    # --- paths ---
    if ENV == 'colab':
        # Colab: Google Drive
        ROOT = Path("/content/drive/MyDrive/ScientificNamesNER/SciBert")
    else:
        # Local: parent root
        ROOT = Path(__file__).resolve().parent

    print(f"Environment: {ENV}")
    print(f"Root directory: {ROOT}")

    # Input data
    ANNOTATED_DATA = ROOT.parent/ 'annotated_data'/ 'top_10_medicinal_hits'/ 'annotations'/ 'manually_annotated_chunks'
    LABEL_STUDIO_FILE = ANNOTATED_DATA/ 'task_for_labelstudio_completed_updated.json'
    IDS = ROOT.parent/ 'LLM_models'/ 'evaluating'/ 'outputs'
    TUNING_CSV = IDS / "for_hparam_tuning.csv"
    TESTING_CSV = IDS / "for_testing.csv"

    # Output directories
    OUTPUTS = ROOT/ 'outputs'
    SPLITS = ROOT/ 'splits'
    MODELS = ROOT/ 'models'
    LOGS = ROOT/  'logs'
    PLOTS = ROOT/'outputs'/ 'plots'
    # Create directories
    for dir_path in [OUTPUTS, SPLITS, MODELS, LOGS, PLOTS]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ---- config model ----
    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    MAX_SEQ_LENGTH = 512  # BERT limit

    # CHUNKING STRATEGY
    # - MAX_TOKENS=450: Leaves room for  classification [CLS] and separator [SEP] tokens, and special tokens
    # e.g. run from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    # tokens = tokenizer("Panax ginseng treats headhache")
    # tokens["input_ids"]
    # tokenizer.convert_ids_to_tokens(tokens["input_ids"])
    #['[CLS]', 'P', '##ana', '##x', 'gins', '##eng', 'treats', 'head', '##ha', '##che', '[SEP]']
    # - OVERLAP_SENTS=3: We experimented first with 3 but retained relations < 90%
    MAX_CHUNK_TOKENS = 450
    OVERLAP_SENTENCES = 4

    # ---- Entity types---
    ENTITY_TYPES = [
        "Scientific Plant Name",
        "Scientific Fungus Name",
        "Medicinal Effect",
        "Medical Condition",
    ]

    # BIO tagging scheme
    # Why BIO? Distinguishes entity boundaries (B=Begin, I=Inside)
    # Example: "Panax ginseng" -> [B-Scientific Plant Name, I-Scientific Plant Name]
    # See https://huggingface.co/docs/transformers/tasks/token_classification

    BIO_LABELS = ["O"]  # O = Outside (not an entity)
    for entity_type in ENTITY_TYPES:
        BIO_LABELS.extend([f"B-{entity_type}", f"I-{entity_type}"])
    # This creats a map from labels to id and viceversa see https://github.com/huggingface/course/blob/main/chapters/en/chapter7/2.mdx
    LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
    ID2LABEL = {i: label for label, i in LABEL2ID.items()}

    # --- Relation types ---
    # Entity constraints or valid pairs
    #   - (Scientific Plant/Fungus Name) -> (Medicinal Effect)
    #   - (Scientific Plant/Fungus Name) -> (Medical Condition)
    RELATION_TYPES = [
        "no_relation",
        "has_medicinal_effect",
        "treats_medical_condition"
    ]
    # This creates a map from labels to id and viceversa
    REL_LABEL2ID = {label: i for i, label in enumerate(RELATION_TYPES)}
    REL_ID2LABEL = {i: label for label, i in REL_LABEL2ID.items()}

    @staticmethod
    def valid_relation_types(head_type: str, tail_type: str) -> set:
        """
        Returns the set of relation labels valid for a head/tail entity type pair.
        Scientific Name + Medicinal Effect  -> {has_medicinal_effect, no_relation}
        Scientific Name + Medical Condition -> {treats_medical_condition, no_relation}
        Any other combination               -> {} (skip entirely)
        Fix: avoid SciName -> treats_medical_condition -> Medicinal effect
        This is useful for train REL script
        """
        scientific_names = {'Scientific Plant Name', 'Scientific Fungus Name'}
        if head_type not in scientific_names:
            return set()
        if tail_type == 'Medicinal Effect':
            return {'has_medicinal_effect', 'no_relation'}
        if tail_type == 'Medical Condition':
            return {'treats_medical_condition', 'no_relation'}
        return set()


    # ---- Trainig Hp ---


    NER_RE_FULL = {
        # Values from LoRA: Low-Rank Adaptation of Large Language Models.
        # https://arxiv.org/abs/2106.09685 # See table 11, 10, 9 for learning rate, LoRA r, alpha and dropout
        "learning_rate": 3e-4,
        "batch_size": 16, # number of training sample propagated. In future try higher values
        "num_epochs": 10, # table 3 in Mosbach et al 2021 https://arxiv.org/pdf/2006.04884
        "lora_r": 16, # LoRA rank controlos number of training parameters in LoRA adapter matrices.
        "lora_alpha": 32, # LORA alpha: scales the adapter update by alpha/r. In this case alpha = 2r
        "lora_dropout": 0.1, # regularization techniques
        "max_length": 512, # Max sequence length based on limit of BERT-based models.
        "warmup_ratio": 0.1,#table 3 in Mosbach et al 2021
        "weight_decay": 0.01, # L2 regularization (like in ridge regression) Mosbach et al. 2021 uses λ=0.01
        "negative_sample_ratio": 2,  # no_relation : positive pairs. 12:1 natural ratio (see output re_pairs_stats.csv).
        # 2:1 sampled used here and only in RE
    }


    # ---- Cross Val ----
    N_FOLDS = 5

    # --- Vals ---
    @classmethod
    def validate(cls):
        """Check that required files exist"""
        required_files = [
            cls.LABEL_STUDIO_FILE,
            cls.TUNING_CSV,
            cls.TESTING_CSV,
        ]

        missing = [f for f in required_files if not f.exists()]
        if missing:
            print("\n Missing required files:")
            for f in missing:
                print(f"   - {f}")
            raise FileNotFoundError("Please ensure all required data files are present")

        print("All required files found")
        return True

    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Environment: {cls.ENV}")
        print(f"Root directory: {cls.ROOT}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Entity types: {len(cls.ENTITY_TYPES)}")
        print(f"BIO labels: {len(cls.BIO_LABELS)}")
        print(f"Relation types: {len(cls.RELATION_TYPES)}")
        print(f"Cross-validation folds: {cls.N_FOLDS}")
        print("=" * 60 + "\n")


# Auto-run validation and summary when imported
if __name__ == "__main__":
    Config.validate()
    Config.print_summary()