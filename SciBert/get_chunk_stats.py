# To run this needed to disinstall torch because structured_output_schema was loading langchain
import json
import pandas as pd
from pathlib import Path
from SciBert.config import Config

from LLM_models.structured_output_schema import (
    summarise_annotations,
    get_all_human_annotations_for_chunk_id,
    deduplicate_and_standardise_output_taxa_lists,
    Taxon,
    TaxaData,
)
from useful_string_methods import TAXON_ENTITY_CLASSES


def get_all_human_annotations_for_scibert_chunk(scibert_chunk: dict, standardise_annotations: bool = True) -> TaxaData: # standardise true allow fair comparison
    entities = {i: e for i, e in enumerate(scibert_chunk['entities'])}
    collected_output = {}

    for rel in scibert_chunk['relations']:
        head = entities[rel['head']]
        tail = entities[rel['tail']]
        if head['label'] in TAXON_ENTITY_CLASSES:
            sci = head['text']
            if sci not in collected_output:
                collected_output[sci] = {'medical_conditions': [], 'medicinal_effects': []}
            if rel['label'] == 'has_medicinal_effect':
                collected_output[sci]['medicinal_effects'].append(tail['text'])
            elif rel['label'] == 'treats_medical_condition':
                collected_output[sci]['medical_conditions'].append(tail['text'])

    for e in scibert_chunk['entities']:
        if e['label'] in TAXON_ENTITY_CLASSES:
            if e['text'] not in collected_output:
                collected_output[e['text']] = {'medical_conditions': [], 'medicinal_effects': []}

    collected_taxa_data = []
    for sci, vals in collected_output.items():
        collected_taxa_data.append(Taxon(
            scientific_name=sci,
            medical_conditions=vals['medical_conditions'] or None,
            medicinal_effects=vals['medicinal_effects'] or None,
        ))

    if standardise_annotations:
        return deduplicate_and_standardise_output_taxa_lists(collected_taxa_data)
    else:
        return TaxaData(taxa=collected_taxa_data)


def get_sets_from_chunk_ids(chunk_ids):
    sci, eff, cond = set(), set(), set()
    for chunk_id in chunk_ids:
        for t in get_all_human_annotations_for_chunk_id(chunk_id, check=False).taxa:
            if t.scientific_name:
                sci.add(t.scientific_name)
                for e in t.medicinal_effects or []: eff.add(f'{t.scientific_name}_{e}')
                for c in t.medical_conditions or []: cond.add(f'{t.scientific_name}_{c}')
    return sci, eff, cond


def get_sets_from_jsonl(path):
    sci, eff, cond = [], [], []
    with open(path) as f:
        for line in f:
            for t in get_all_human_annotations_for_scibert_chunk(json.loads(line)).taxa:
                if t.scientific_name:
                    sci.append(t.scientific_name)
                    for e in t.medicinal_effects or []: eff.append(f'{t.scientific_name}_{e}')
                    for c in t.medical_conditions or []: cond.append(f'{t.scientific_name}_{c}')
    return sci, eff, cond


def main():
    out_dir = Config.OUTPUTS / "chunks_stat"
    out_dir.mkdir(exist_ok=True)

    val_ids  = pd.read_csv(Config.TUNING_CSV)["id"].tolist()
    test_ids = pd.read_csv(Config.TESTING_CSV)["id"].tolist()

    # Part 1: val, test, full summaries
    val_taxa,  val_conds,  val_effects  = summarise_annotations(val_ids,            out_dir / "summary_val.csv")
    test_taxa, test_conds, test_effects = summarise_annotations(test_ids,           out_dir / "summary_test.csv")
    summarise_annotations(val_ids + test_ids,                                       out_dir / "summary_full.csv")

    # Overlap val vs test
    _, val_eff_pairs,  val_cond_pairs  = get_sets_from_chunk_ids(val_ids)
    _, test_eff_pairs, test_cond_pairs = get_sets_from_chunk_ids(test_ids)

    pd.DataFrame([{
        'common_scinames': len(val_taxa  & test_taxa),
        'common_med_eff':  len(val_eff_pairs  & test_eff_pairs),
        'common_med_cond': len(val_cond_pairs & test_cond_pairs),
    }]).to_csv(out_dir / "overlap_val_test.csv", index=False)

    # Part 2: per-fold CV
    comb_sci, comb_eff, comb_cond = set(), set(), set()
    fold_rows = []

    for fold in range(1, Config.N_FOLDS + 1):
        for label, path in [
            ("train", Config.OUTPUTS / f"train_chunks_fold{fold}.jsonl"),
            ("test",  Config.OUTPUTS / f"test_chunks_fold{fold}.jsonl"),
        ]:
            sci, eff, cond = get_sets_from_jsonl(path)
            fold_rows.append({
                'fold': fold, 'split': label,
                'scinames_total': len(sci),   'scinames_unique': len(set(sci)),
                'med_eff_total':  len(eff),   'med_eff_unique':  len(set(eff)),
                'med_cond_total': len(cond),  'med_cond_unique': len(set(cond)),
            })
            if label == "test":
                comb_sci  |= set(sci)
                comb_eff  |= set(eff)
                comb_cond |= set(cond)

    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_fold_stats.csv", index=False)

    # Part 3: combined test folds vs original test set
    pd.DataFrame([{
        'combined_scinames': len(comb_sci),  'original_scinames': len(test_taxa),  'lost_scinames': len(test_taxa  - comb_sci),
        'combined_med_eff':  len(comb_eff),  'original_med_eff':  len(test_effects),'lost_med_eff':  len(test_effects - comb_eff),
        'combined_med_cond': len(comb_cond), 'original_med_cond': len(test_conds), 'lost_med_cond': len(test_conds  - comb_cond),
    }]).to_csv(out_dir / "combined_folds_vs_test.csv", index=False)


if __name__ == "__main__":

    main()