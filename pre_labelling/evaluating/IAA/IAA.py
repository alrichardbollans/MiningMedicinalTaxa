import json
import pandas as pd
from typing import List

# Import your required functions and classes
from LLM_models.structured_output_schema import (
    get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations,
    convert_human_annotations_to_taxa_data_schema,
    deduplicate_and_standardise_output_taxa_lists
)
from LLM_models.evaluating import (
    NER_evaluation,
    RE_evaluation,
    get_metrics_from_tp_fp_fn,
    abbreviated_precise_match,
    abbreviated_approximate_match
)

# Path to your annotation file
_annotation_file = "iaa.json"


# Load annotations for 2 annotators from a single data entry
def load_annotations_for_two_annotators(chunk_entry: dict):
    anns_list = [a["result"] for a in chunk_entry.get("annotations", [])]

    # If fewer than 2 annotators, skip this entry
    if len(anns_list) < 2:
        return None, None

    try:
        # Separate NER and RE annotations for annotator 1
        human_ner_1, human_re_1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[0], check=True, standardise_annotations=True
        )

        # Separate NER and RE annotations for annotator 2
        human_ner_2, human_re_2 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[1], check=True, standardise_annotations=True
        )

        # Convert human annotations to TaxaData objects (structured format)
        taxa_data_1 = convert_human_annotations_to_taxa_data_schema(human_ner_1, human_re_1)
        taxa_data_2 = convert_human_annotations_to_taxa_data_schema(human_ner_2, human_re_2)

        return taxa_data_1, taxa_data_2

    except Exception as e:
        print(f"Error processing chunk {chunk_entry['id']}: {e}")
        return None, None


# Main execution starts here
if __name__ == "__main__":

    # Load JSON data containing annotated chunks
    with open(_annotation_file) as f:
        data = json.load(f)

    all_annotator1 = []  # Will hold TaxaData from annotator 1 across all chunks
    all_annotator2 = []  # Will hold TaxaData from annotator 2 across all chunks

    # Process each annotated chunk
    for entry in data:
        a1, a2 = load_annotations_for_two_annotators(entry)

        if a1 is None or a2 is None:
            continue  # Skip if invalid or incomplete annotations

        all_annotator1.append(a1)
        all_annotator2.append(a2)

    # Flatten Taxon objects into raw lists for both annotators
    all_taxa_annotator1 = []
    all_taxa_annotator2 = []

    for td in all_annotator1:
        all_taxa_annotator1.extend(td.taxa)

    for td in all_annotator2:
        all_taxa_annotator2.extend(td.taxa)

    # Apply deduplication and standardisation of taxa lists
    cleaned_a1 = deduplicate_and_standardise_output_taxa_lists(all_taxa_annotator1)
    cleaned_a2 = deduplicate_and_standardise_output_taxa_lists(all_taxa_annotator2)

    # ----------------------------------------------------------------------------------------
    print("\n=== NER Evaluation: Scientific Names Only ===")

    # NER Evaluation: Compare scientific names between annotators
    tp_g1, tp_m2, fp2, fn1 = NER_evaluation(cleaned_a2, cleaned_a1, abbreviated_precise_match)
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Exact match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = NER_evaluation(cleaned_a1, cleaned_a2, abbreviated_precise_match)
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nExact match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

    # NER Evaluation: Compare scientific names between annotators
    tp_g1, tp_m2, fp2, fn1 = NER_evaluation(cleaned_a2, cleaned_a1, abbreviated_approximate_match)
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Approximate match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = NER_evaluation(cleaned_a1, cleaned_a2, abbreviated_approximate_match)
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nApproximate match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

    # ----------------------------------------------------------------------------------------
    print("\n=== NERRE Evaluation: Scientific Names - Medical Condition ===")
    # RE evaluation
    tp_g1, tp_m2, fp2, fn1 = RE_evaluation(cleaned_a2, cleaned_a1, 'precise', 'medical_conditions' )
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Precise Match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = RE_evaluation(cleaned_a1, cleaned_a2, 'precise', 'medical_conditions')
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nPrecise Match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

    # RE evaluation
    tp_g1, tp_m2, fp2, fn1 = RE_evaluation(cleaned_a2, cleaned_a1, 'approximate', 'medical_conditions')
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Approximate Match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = RE_evaluation(cleaned_a1, cleaned_a2, 'approximate', 'medical_conditions')
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nApproximate Match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

    # ----------------------------------------------------------------------------------------
    print("\n=== NERRE Evaluation: Scientific Names - Medicinal Effect ===")
    # RE evaluation
    tp_g1, tp_m2, fp2, fn1 = RE_evaluation(cleaned_a2, cleaned_a1, 'precise', 'medicinal_effects' )
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Precise Match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = RE_evaluation(cleaned_a1, cleaned_a2, 'precise', 'medicinal_effects')
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nPrecise Match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")

    # RE evaluation
    tp_g1, tp_m2, fp2, fn1 = RE_evaluation(cleaned_a2, cleaned_a1, 'approximate', 'medicinal_effects' )
    p1, r1, f1_1 = get_metrics_from_tp_fp_fn(tp_g1, tp_m2, fp2, fn1)
    print(f"Approximate Match: Annotator 1 as Ground Truth → Annotator 2 evaluated")
    print(f"Precision: {p1:.3f}, Recall: {r1:.3f}, F1: {f1_1:.3f}")

    tp_g2, tp_m1, fp1, fn2 = RE_evaluation(cleaned_a1, cleaned_a2, 'approximate', 'medicinal_effects')
    p2, r2, f1_2 = get_metrics_from_tp_fp_fn(tp_g2, tp_m1, fp1, fn2)
    print(f"\nApproximate Match: Annotator 2 as Ground Truth → Annotator 1 evaluated")
    print(f"Precision: {p2:.3f}, Recall: {r2:.3f}, F1: {f1_2:.3f}")


    #--------------------------------------------------------------------------------------------
    print("\n=== Data Preview: Annotator 2 Cleaned Taxa ===")

    # Create a DataFrame of Annotator 2's cleaned taxa for inspection
    df2 = pd.DataFrame([t.dict() for t in cleaned_a2.taxa])
    print(df2.head(10))
