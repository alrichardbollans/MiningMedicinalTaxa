import itertools
import os.path
import string
from typing import Callable

import numpy as np
import pandas as pd

from rag_models.structured_output_schema import TaxaData

def get_metrics_from_tp_fp_fn(true_positives_in_ground_truths: list, true_positives_in_model_annotations: list, false_positives: list,
                              false_negatives: list):
    """
    Calculate precision, recall, and F1 score given lists of true positives (TP), false positives (FP),
    and false negatives (FN).

    :param true_positives: The number of true positives.
    :param false_positives: The number of false positives.
    :param false_negatives: The number of false negatives.
    :return: A tuple containing the precision, recall, and F1 score.
    """
    # precision
    if len(true_positives_in_model_annotations + false_positives) != 0:
        precision = len(true_positives_in_model_annotations) / len(true_positives_in_model_annotations + false_positives)
    else:
        print(f'Zero Division WARNING: No positive annotations given by model.')
        print(f'Setting precision to np.nan')
        precision = np.nan
    # recall
    if len(true_positives_in_ground_truths + false_negatives) != 0:
        recall = len(true_positives_in_ground_truths) / len(true_positives_in_ground_truths + false_negatives)
    else:
        print(f'Zero Division WARNING: No positive ground truth annotations.')
        print(f'Setting recall to np.nan')
        recall = np.nan
    # f1
    # denom = ((2 * len(true_positives)) + len(false_positives) + len(false_negatives))
    if recall == recall and precision == precision and precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        print(f'Zero Division WARNING: No TP,FP,FN.')
        print(f'Setting f1_score to np.nan')
        f1_score = np.nan
    return precision, recall, f1_score

def abbreviate(name1: str) -> str:
    """
    Return given name with first word abbreviated, if there are multiple words.
    :param name1:
    :return:
    """
    if '  ' in name1:
        raise ValueError(f'Double spacing found in name: {name1}. This will break abbreviation formating.')
    words = name1.split()
    if len(words) < 2:
        return name1
    else:
        words[0] = words[0][0] + '.'
        return ' '.join(words)


def abbreviated_precise_match(name1: str, name2: str):
    if name1 == name2 or abbreviate(name1) == name2 or abbreviate(name2) == name1:
        return True
    else:
        return False


def abbreviated_approximate_match(name1: str, name2: str):
    if approximate_match(name1, name2) or approximate_match(abbreviate(name1), name2) or approximate_match(abbreviate(name2), name1):
        return True
    else:
        return False


def precise_match(name1: str, name2: str):
    """

    """
    if name1.lower() != name1.lower() or name2.lower() != name2.lower():
        raise ValueError(f'Names not lower case: {name1}, {name2}')

    if name1 == name2:
        return True
    else:
        return False


def approximate_match(name1: str, name2: str):
    """

    """
    name1_split = [word.strip(string.punctuation) for word in name1.split()]
    name2_split = [word.strip(string.punctuation) for word in name2.split()]

    def get_succesive_combinations_of_words(namelist):
        possible_splits = []
        for i in range(len(namelist)):
            # if i > 0:
            #     for j in range(i):
            #         possible_splits.append(' '.join(namelist[j:i + 1]))
            #     possible_splits.append(namelist[i])
            # else:
            #     possible_splits.append(' '.join(namelist[:i + 1]))
            possible_splits.append(' '.join(namelist[:i + 1]))
        return possible_splits

    possible_2_splits = get_succesive_combinations_of_words(name2_split)
    possible_1_splits = get_succesive_combinations_of_words(name1_split)

    cleaned_name1 = ' '.join(name1_split)
    cleaned_name2 = ' '.join(name2_split)
    if cleaned_name1 in possible_2_splits or cleaned_name2 in possible_1_splits:
        return True
    else:
        return False


def NER_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: Callable):
    """

    """
    ground_truth_names = [g.scientific_name for g in ground_truth_annotations.taxa]
    assert len(ground_truth_names) == len(ground_truth_annotations.taxa)
    assert len(ground_truth_names) == len(set(ground_truth_names))  # These should have been deduplicated

    true_positives_in_ground_truths = []
    true_positives_in_model_annotations = []
    # Collect true positives
    # When calculating recall, want the true positives with respect to false negatives i.e. the number of ground truth scientific names that have been matched
    # When calculating precision, want the true positives with respect to false positives i.e. the number of predicted scientific names that have been matched
    # Usually these two things would be the same, but when using an approximate matching method you might can end up with unwanted duplications
    for a in model_annotations.taxa:
        for g in ground_truth_annotations.taxa:
            if matching_method(a.scientific_name, g.scientific_name):
                if g.scientific_name not in true_positives_in_ground_truths:
                    true_positives_in_ground_truths.append(g.scientific_name)
                if a.scientific_name not in true_positives_in_model_annotations:
                    true_positives_in_model_annotations.append(a.scientific_name)

    # False positives
    false_positives = []
    for a in model_annotations.taxa:
        if not (any(matching_method(a.scientific_name, g.scientific_name) for g in ground_truth_annotations.taxa)):
            false_positives.append(a.scientific_name)

    # False negatives
    false_negatives = []
    for g in ground_truth_annotations.taxa:
        if not (any(matching_method(g.scientific_name, a.scientific_name) for a in model_annotations.taxa)):
            false_negatives.append(g.scientific_name)

    assert len(ground_truth_names) == len(true_positives_in_ground_truths + false_negatives)
    print(f'False positives: {false_positives}')
    print(f'False negatives: {false_negatives}')
    return true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives


def RE_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: str, relationship: str):
    # TODO: check this

    if matching_method == 'precise':
        NER_matching_method = abbreviated_precise_match
        RE_matching_method = precise_match
    elif matching_method == 'approximate':
        NER_matching_method = abbreviated_approximate_match
        RE_matching_method = approximate_match
    else:
        raise ValueError(f'Unrecognised matching method: {matching_method}')

    true_positives_in_ground_truths = []
    true_positives_in_model_annotations = []
    for model_ann in model_annotations.taxa:
        for model_med_cond in getattr(model_ann, relationship) or []:
            for g in ground_truth_annotations.taxa:
                if NER_matching_method(model_ann.scientific_name, g.scientific_name):
                    for ground_med_condition in getattr(g, relationship) or []:
                        if RE_matching_method(model_med_cond, ground_med_condition):
                            model_string = '_'.join([model_ann.scientific_name, model_med_cond])
                            if model_string not in true_positives_in_model_annotations:
                                true_positives_in_model_annotations.append(model_string)

                            ground_truth_string = '_'.join([g.scientific_name, ground_med_condition])
                            if ground_truth_string not in true_positives_in_ground_truths:
                                true_positives_in_ground_truths.append(ground_truth_string)

    false_positives = []
    for model_ann in model_annotations.taxa:
        for model_med_cond in getattr(model_ann, relationship) or []:
            model_string = '_'.join([model_ann.scientific_name, model_med_cond])
            if not (any(NER_matching_method(model_ann.scientific_name, g.scientific_name) for g in ground_truth_annotations.taxa)):
                # If no matching scientific name then this is a false positive
                false_positives.append(model_string)
            else:
                found = False
                for g in ground_truth_annotations.taxa:
                    if NER_matching_method(model_ann.scientific_name, g.scientific_name):
                        for ground_med_condition in getattr(g, relationship) or []:
                            if RE_matching_method(model_med_cond, ground_med_condition):
                                found = True
                if not found:
                    false_positives.append(model_string)
    assert len(false_positives) == len(set(false_positives))

    # False negatives
    false_negatives = []
    for g in ground_truth_annotations.taxa:
        for gt_med_cond in getattr(g, relationship) or []:
            ground_truth_string = '_'.join([g.scientific_name, gt_med_cond])
            if not (any(NER_matching_method(g.scientific_name, a.scientific_name) for a in model_annotations.taxa)):
                false_negatives.append(ground_truth_string)
            else:
                found = False
                for m in model_annotations.taxa:
                    if NER_matching_method(m.scientific_name, g.scientific_name):
                        for m_med_condition in getattr(m, relationship) or []:
                            if RE_matching_method(m_med_condition, gt_med_cond):
                                found = True
                if not found:
                    false_negatives.append(ground_truth_string)
    assert len(false_negatives) == len(set(false_negatives))
    print(f'False positives: {false_positives}')
    print(f'False negatives: {false_negatives}')
    return true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives


def check_errors(model_annotations: TaxaData, ground_truth_annotations: TaxaData, out_dir: str, chunk_id: int, model_tag: str):
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
    all = [true_positives_in_model_annotations, false_positives, false_negatives,
           retrue_positives_in_model_annotations, refalse_positives,
           refalse_negatives, meretrue_positives_in_model_annotations, merefalse_positives, merefalse_negatives]
    padded = list(zip(*itertools.zip_longest(*all, fillvalue='')))
    problems = pd.DataFrame(zip(*padded),
                            columns=['NER_tp', 'NER_fp', 'NER_fn', 'MedCond_tp', 'MedCond_fp', 'MedCond_fn', 'MedEff_tp', 'MedEff_fp', 'MedEff_fn'])
    problems.to_csv(os.path.join(out_dir, f'{str(chunk_id)}_{model_tag}_problems.csv'))


def clean_model_annotations_using_taxonomy_knowledge(model_annotations: TaxaData):
    ## Set up as an assessment of performance when we autoremove names that don't appear in taxonomic lists (this will better reflect usage).
    new_taxa_list = []
    kword_dict = get_kword_dict()
    for model_ann in model_annotations.taxa:
        if any(approximate_match(model_ann.scientific_name, x) for x in all_names):
            new_taxa_list.append(model_ann)

    return TaxaData(taxa=new_taxa_list)
