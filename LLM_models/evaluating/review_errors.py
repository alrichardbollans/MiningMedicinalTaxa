import itertools
import os
import pickle
from difflib import SequenceMatcher

import pandas as pd

from LLM_models.evaluating import abbreviated_precise_match, RE_evaluation, NER_evaluation, abbreviated_approximate_match
from LLM_models.structured_output_schema import get_all_human_annotations_for_chunk_id

def get_re_fp_fn(model_name: str, chunk_id):
    ground_truth_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
    pkl_file = os.path.join('outputs', 'model_pkls', f'{str(chunk_id)}_{model_name}_outputs.pickle')
    model_annotations = pickle.load(open(pkl_file, "rb", -1))

    retrue_positives_in_ground_truths, retrue_positives_in_model_annotations, refalse_positives, refalse_negatives = RE_evaluation(model_annotations,
                                                                                                                                   ground_truth_annotations,
                                                                                                                                   'approximate',
                                                                                                                                   'medical_conditions')
    meretrue_positives_in_ground_truths, meretrue_positives_in_model_annotations, merefalse_positives, merefalse_negatives = RE_evaluation(
        model_annotations,
        ground_truth_annotations,
        'approximate',
        'medicinal_effects')
    return merefalse_positives, merefalse_negatives, refalse_positives, refalse_negatives, model_annotations, ground_truth_annotations


def manually_compare_errors(model_name: str, chunk_id, out_dir: str):
    ground_truth_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
    pkl_file = os.path.join('outputs', 'model_pkls', f'{str(chunk_id)}_{model_name}_outputs.pickle')
    model_annotations = pickle.load(open(pkl_file, "rb", -1))

    true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives = NER_evaluation(model_annotations,
                                                                                                                            ground_truth_annotations,
                                                                                                                            abbreviated_approximate_match)
    retrue_positives_in_ground_truths, retrue_positives_in_model_annotations, refalse_positives, refalse_negatives = RE_evaluation(model_annotations,
                                                                                                                                   ground_truth_annotations,
                                                                                                                                   'approximate',
                                                                                                                                   'medical_conditions')
    meretrue_positives_in_ground_truths, meretrue_positives_in_model_annotations, merefalse_positives, merefalse_negatives = RE_evaluation(
        model_annotations,
        ground_truth_annotations,
        'approximate',
        'medicinal_effects')
    if len(merefalse_positives)>0:
            print(merefalse_positives)
    all = [true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives,
           retrue_positives_in_model_annotations, refalse_positives,
           refalse_negatives, meretrue_positives_in_model_annotations, merefalse_positives, merefalse_negatives]
    padded = list(zip(*itertools.zip_longest(*all, fillvalue='')))
    problems = pd.DataFrame(zip(*padded),
                            columns=['NER_tp_in_ground', 'NER_tp_in_model', 'NER_fp', 'NER_fn', 'MedCond_tp', 'MedCond_fp', 'MedCond_fn', 'MedEff_tp',
                                     'MedEff_fp', 'MedEff_fn'])



    ### compare with
    precise_NER_true_positives_in_ground_truths, precise_NER_true_positives_in_model_annotations, precise_NER_false_positives, precise_NER_false_negatives = NER_evaluation(
        model_annotations,
        ground_truth_annotations,
        abbreviated_precise_match)

    NER_fps = [c for c in precise_NER_false_positives if c not in false_positives]
    NER_fns = [c for c in precise_NER_false_negatives if c not in false_negatives]

    (preciseMedCondtrue_positives_in_ground_truths, preciseMedCondtrue_positives_in_model_annotations, preciseMedCondfalse_positives,
     preciseMedCondfalse_negatives) = RE_evaluation(model_annotations,
                                                    ground_truth_annotations,
                                                    'precise',
                                                    'medical_conditions')

    MC_fps = [c for c in preciseMedCondfalse_positives if c not in refalse_positives]
    MC_fns = [c for c in preciseMedCondfalse_negatives if c not in refalse_negatives]

    (preciseMedEfftrue_positives_in_ground_truths, preciseMedEfftrue_positives_in_model_annotations, preciseMedEfffalse_positives,
         preciseMedEfffalse_negatives) = RE_evaluation(model_annotations,
                                                       ground_truth_annotations,
                                                       'precise',
                                                       'medicinal_effects')
    ME_fps = [c for c in preciseMedEfffalse_positives if c not in merefalse_positives]
    ME_fns = [c for c in preciseMedEfffalse_negatives if c not in merefalse_negatives]
    important_cases = [MC_fps,MC_fns, ME_fps, ME_fns]
    if any(len(x)>0 for x in important_cases):
        print(f'chunk: {chunk_id}')
        print(f'model: {model_name}')
        print(MC_fps)


def check_for_spelling(model):
    med_eff_cases = []
    medCond_cases = []
    for chunk in test['id'].unique().tolist():
        medEff_false_positives, medEff_false_negatives, medCond_false_positives, medCond_false_negatives, model_annotations, ground_truth_annotations =get_re_fp_fn(model, chunk)
        for a in medCond_false_negatives:
            for b in medCond_false_positives:
                ratio = SequenceMatcher(None, a, b).ratio()
                if ratio > 0.8:
                    medCond_cases.append((a, b))
        for a in medEff_false_negatives:
            for b in medEff_false_positives:
                ratio = SequenceMatcher(None, a, b).ratio()
                if ratio > 0.8:
                    med_eff_cases.append((a, b))
    medcond_df = pd.DataFrame(medCond_cases, columns=['ground truth', 'model output'])
    medcond_df.to_csv(os.path.join('outputs', 'possible_spelling_mistakes', f'{model}_medcond.csv'))

    medEff_df = pd.DataFrame(med_eff_cases, columns=['ground truth', 'model output'])
    medEff_df.to_csv(os.path.join('outputs', 'possible_spelling_mistakes', f'{model}_medEff.csv'))


def compare_outputs():
    for chunk in test['id'].unique().tolist():
        manually_compare_errors('gpt-4o-2024-08-06', chunk, os.path.join('outputs', 'full_eval', 'comparing exact and relaxed'))


if __name__ == '__main__':
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    # compare_outputs()
    check_for_spelling('gpt-4o-2024-08-06')