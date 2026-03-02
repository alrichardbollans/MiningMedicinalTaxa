#!/usr/bin/env python3
# 06_evaluate_scibert.py
"""
Aggregate SciBERT predictions across all 5 folds and evaluate against ground truth.

ROLE IN PIPELINE:
    e_prediction produces predictions_fold{N}_taxadata.json for each fold.
    Each fold covers a set of chunks (task_ids). The names and relationships are merged
    and deduplicated

    Predictions are aggregated to chunk level (task_id) using
    deduplicate_and_standardise_output_taxa_lists,
    then evaluated via assess_model_on_chunk_list
    from run_evaluation.py (same function used for all other baselines).
    Output CSV is in the standard format so make_nice_plots.py can include
    SciBERT alongside Claude, GPT, TaxoNERD etc.

    NOTE: to add SciBERT to make_nice_plots.py, add this line to the renaming
    dict in get_filenames():
        'scibert_results.csv': 'SciBERT',

PREREQUISITES:
    - 05_end_to_end_prediction.py completed for all folds
    - KEWSCRATCHPATH environment variable set

OUTPUT:
    outputs/full_eval/scibert_results.csv
    outputs/full_eval/scibert_results.png

USAGE:
    python 06_evaluate_scibert.py
"""

import json
import os
import pickle
from typing import Dict, List

from SciBert.config import Config
from LLM_models.structured_output_schema import Taxon, TaxaData, deduplicate_and_standardise_output_taxa_lists
from LLM_models.evaluating.run_evaluation import assess_model_on_chunk_list


# ---------------------------------------------------------------------------
# Dummy model — required by assess_model_on_chunk_list interface
# ---------------------------------------------------------------------------

class _ModelTag:
    """Minimal object satisfying the model_name interface of assess_model_on_chunk_list."""
    model_name = 'scibert'


# ---------------------------------------------------------------------------
# Aggregate predictions across folds
# ---------------------------------------------------------------------------

def aggregate_predictions() -> Dict[int, TaxaData]:
    """
    Load predictions from all 5 folds, group task_id (chunk_id)
    All Taxon objects for the same chunk are collected into a flat list and
    passed to deduplicate_and_standardise_output_taxa_lists.
    """
    by_task: Dict[int, List[Taxon]] = {}

    for fold in range(1, Config.N_FOLDS + 1):
        pred_file = Config.OUTPUTS / f"predictions_fold{fold}_taxadata.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"Run 05 first: {pred_file}")

        with open(pred_file, encoding='utf-8') as f:
            predictions = json.load(f)

        for entry in predictions:
            task_id  = entry['task_id']
            taxadata = TaxaData(**entry['taxadata'])
            if task_id not in by_task:
                by_task[task_id] = []
            for taxon in taxadata.taxa or []:
                by_task[task_id].append(taxon)

    return {
        task_id: deduplicate_and_standardise_output_taxa_lists(taxa)
        for task_id, taxa in by_task.items()
    }


# --- run ----

def main():
    Config.validate()
    out_dir    = Config.IDS / 'full_eval'
    pkl_dir    = Config.IDS / 'model_pkls'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(os.path.join(Config.IDS, 'model_errors'), exist_ok=True)

    print("Aggregating predictions across all folds...")
    predictions = aggregate_predictions()
    print(f"  {len(predictions)} documents aggregated.")

    # Dump each document's TaxaData to pickle so assess_model_on_chunk_list
    # can load it with rerun=False — avoids rewriting the evaluation loop.
    print("Writing prediction pickles...")
    for task_id, taxadata in predictions.items():
        pkl_path = os.path.join(pkl_dir, f"{task_id}_scibert_outputs.pickle")
        with open(pkl_path, 'wb') as f:
            pickle.dump(taxadata, f)

    task_ids = sorted(predictions.keys())
    model    = _ModelTag()

    print(f"Evaluating {len(task_ids)} documents...")
    assess_model_on_chunk_list(
        task_ids,
        model,
        context_window=None,
        out_dir=out_dir,
        rerun=False,
        autoremove_non_sci_names=False,
    )

    print(f"Evaluating with autoremove_non_sci_names...")
    assess_model_on_chunk_list(
        task_ids,
        model,
        context_window=None,
        out_dir=out_dir,
        rerun=False,
        autoremove_non_sci_names=True,
    )

    print(f"\nSaved: {os.path.join(out_dir, 'scibert_results.csv')}")
    print(f"Saved: {os.path.join(out_dir, 'scibert_results.png')}")
    print("    'scibert_results.csv': 'SciBERT',")


if __name__ == "__main__":
    main()
