import json
import os
import pandas as pd
import numpy as np
from typing import List

from LLM_models.structured_output_schema import (
    get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations,
    convert_human_annotations_to_taxa_data_schema,
    TaxaData,
    Taxon)

from LLM_models.evaluating import (
    NER_evaluation,
    RE_evaluation,
    get_metrics_from_tp_fp_fn,
    abbreviated_precise_match,
    abbreviated_approximate_match)


annotation_file = os.path.join(os.path.dirname(__file__), "iaa.json")
#annotation_file = os.path.join(os.getcwd(), "pre_labelling", "evaluating", "IAA", annotation_file)
def load_annotations_for_two_annotators(chunk_entry: dict):
    anns_list = [a["result"] for a in chunk_entry.get("annotations", [])]
    if len(anns_list) < 2:
        return None, None
    try:
        ner_1, re_1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[0], check=False, standardise_annotations=True)
        ner_2, re_2 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(
            anns_list[1], check=False, standardise_annotations=True)

        taxa_data_1 = convert_human_annotations_to_taxa_data_schema(ner_1, re_1)
        taxa_data_2 = convert_human_annotations_to_taxa_data_schema(ner_2, re_2)
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

#-------------
# run
#-------------

with open(annotation_file, encoding="utf-8") as f:
    data = json.load(f)

all_annotator1 = []
all_annotator2 = []

for entry in data:
    a1, a2 = load_annotations_for_two_annotators(entry)
    if a1 is None or a2 is None:
        continue
    all_annotator1.append(a1)
    all_annotator2.append(a2)

evaluation_plan = [
    ("NER: Scientific Names", "Exact", NER_evaluation, (abbreviated_precise_match,)),
    ("NER: Scientific Names", "Approximate", NER_evaluation, (abbreviated_approximate_match,)),
    ("NERRE: Medical Condition", "Exact", RE_evaluation, ("precise", "medical_conditions")),
    ("NERRE: Medical Condition", "Approximate", RE_evaluation, ("approximate", "medical_conditions")),
    ("NERRE: Medicinal Effect", "Exact", RE_evaluation, ("precise", "medicinal_effects")),
    ("NERRE: Medicinal Effect", "Approximate", RE_evaluation, ("approximate", "medicinal_effects")),
]

print("\n=== Bootstrap CI (1000 iterations, 95%) ===")

rng = np.random.default_rng(42)
size = len(all_annotator1)
boot_records = []



for _ in range(1000):
    idx = rng.integers(0, size, size=size)
    s1 = merge_taxa_data([all_annotator1[i] for i in idx], deduplicate=True)
    s2 = merge_taxa_data([all_annotator2[i] for i in idx], deduplicate=True)

    for category, match_type, eval_fn, args in evaluation_plan:
        for direction, gt, pred in [
            ("A1 GT to A2 Eval", s1, s2),
            ("A2 GT to A1 Eval", s2, s1),
        ]:
            tp_g, tp_m, fp, fn = eval_fn(gt, pred, *args)
            p, r, f1 = get_metrics_from_tp_fp_fn(tp_g, tp_m, fp, fn)
            boot_records.append(
                {"Category": category, "Match Type": match_type, "Direction": direction,
                 "P": p, "R": r, "F1": f1}
            )

df = pd.DataFrame(boot_records)

summary = (
    df.groupby(["Category", "Match Type", "Direction"])
      .agg(
          P_mean=("P", "mean"),
          P_lo=("P", lambda x: np.percentile(x, 2.5)),
          P_hi=("P", lambda x: np.percentile(x, 97.5)),
          R_mean=("R", "mean"),
          R_lo=("R", lambda x: np.percentile(x, 2.5)),
          R_hi=("R", lambda x: np.percentile(x, 97.5)),
          F1_mean=("F1", "mean"),
          F1_lo=("F1", lambda x: np.percentile(x, 2.5)),
          F1_hi=("F1", lambda x: np.percentile(x, 97.5)),
      )
      .reset_index()
)

summary["P (95% CI)"]  = summary.apply(lambda r: f"{r.P_mean:.3f} [{r.P_lo:.3f}, {r.P_hi:.3f}]", axis=1)
summary["R (95% CI)"]  = summary.apply(lambda r: f"{r.R_mean:.3f} [{r.R_lo:.3f}, {r.R_hi:.3f}]", axis=1)
summary["F1 (95% CI)"] = summary.apply(lambda r: f"{r.F1_mean:.3f} [{r.F1_lo:.3f}, {r.F1_hi:.3f}]", axis=1)

final_df = summary[["Category", "Match Type", "Direction", "P (95% CI)", "R (95% CI)", "F1 (95% CI)"]]

print("\n=== FINAL RESULTS (mean, 95% CI) ===")
print(final_df.to_string(index=False))

out_csv = os.path.join(os.path.dirname(__file__), "results", "evaluation_summary.csv")
final_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")