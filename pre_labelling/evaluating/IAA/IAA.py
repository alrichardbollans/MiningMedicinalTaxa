import json

from LLM_models.structured_output_schema import (
    get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations,
    convert_human_annotations_to_taxa_data_schema,
    TaxaData)
from LLM_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, TaxaData, Taxon
from LLM_models.evaluating import NER_evaluation, get_metrics_from_tp_fp_fn, abbreviated_precise_match

from typing import List

_annotation_file = "iaa.json"

def load_annotations_for_two_annotators(chunk_entry: dict):
    anns_list = [a["result"] for a in chunk_entry.get("annotations", [])]
    if len(anns_list) < 2:
        return None, None
    try:
        human_ner_1, human_re_1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[0], check=True, standardise_annotations=True)
        human_ner_2, human_re_2 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[1], check=True, standardise_annotations=True)

        taxa_data_1 = convert_human_annotations_to_taxa_data_schema(human_ner_1, human_re_1)
        taxa_data_2 = convert_human_annotations_to_taxa_data_schema(human_ner_2, human_re_2)
        return taxa_data_1, taxa_data_2
    except Exception as e:
        print(f"Error processing chunk {chunk_entry['id']}: {e}")
        return None, None


def merge_taxa_data(taxa_data_list: List[TaxaData], deduplicate: bool = True) -> TaxaData:
    all_taxa = []

    for td in taxa_data_list:
        for t in td.taxa:
            if isinstance(t, dict):  # Just in case
                try:
                    all_taxa.append(Taxon(**t))
                except Exception as e:
                    print(f"Skipping invalid taxon: {t} → {e}")
            else:
                all_taxa.append(t)

    if deduplicate:
        # Optional deduplication by scientific name
        unique_taxa = {}
        for t in all_taxa:
            if t.scientific_name not in unique_taxa:
                unique_taxa[t.scientific_name] = t
        all_taxa = list(unique_taxa.values())

    return TaxaData(taxa=all_taxa)

if __name__ == "__main__":
    with open(_annotation_file) as f:
        data = json.load(f)

    all_annotator1 = []
    all_annotator2 = []

    for entry in data:
        a1, a2 = load_annotations_for_two_annotators(entry)
        if a1 is None or a2 is None:
            continue
        all_annotator1.append(a1)
        all_annotator2.append(a2)

    # Merge across all chunks
    merged_a1 = merge_taxa_data(all_annotator1, deduplicate = True)
    merged_a2 = merge_taxa_data(all_annotator2, deduplicate = True)

    print("\n=== Global Metrics (NER, using abbreviated_precise_match) ===")

    # A1 as GT
    tp_g1, tp_m2, fp2, fn1 = NER_evaluation(merged_a2, merged_a1, abbreviated_precise_match)
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    # A2 as GT
    tp_g2, tp_m1, fp1, fn2 = NER_evaluation(merged_a1, merged_a2, abbreviated_precise_match)
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nAnnotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

# get_df
# df2= pd.DataFrame([t.dict() for t in merged_a2.taxa])
# print(df2.head(10))