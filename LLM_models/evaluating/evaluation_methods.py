import itertools
import os.path
import string
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import pandas as pd

from LLM_models.structured_output_schema import TaxaData
from useful_string_methods import filter_name_list_using_sci_names, abbreviate_sci_name

fuzzy_match_ratio = 0.9

def get_metrics_from_tp_fp_fn(true_positives_in_ground_truths: list, true_positives_in_model_annotations: list, false_positives: list,
                              false_negatives: list):
    """
    Calculate precision, recall, and F1 score given lists of true positives (TP), false positives (FP),
    and false negatives (FN).

    :param true_positives_in_model_annotations: The number of true positives within the model annotations.
    :param true_positives_in_ground_truths: The number of true positives within the human annotations.
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
        print(f'Zero Division WARNING: No TP,FP.')
        print(f'Setting f1_score to np.nan')
        f1_score = np.nan
    return precision, recall, f1_score


def abbreviated_precise_match(name1: str, name2: str):
    """
    :param name1: The first name to compare.
    :param name2: The second name to compare.
    :return: True if name1 is an exact match or an abbreviation of name2, or if name2 is an abbreviation of name1. False otherwise.
    """
    if precise_match(name1, name2):
        return True
    if abbreviate_sci_name(name1) == name2 or abbreviate_sci_name(name2) == name1:
        return True
    else:
        return False


def abbreviated_approximate_match(name1: str, name2: str):
    """
    Function to check for an approximate match between two names, including name abbreviations.

    :param name1: The first name to compare.
    :param name2: The second name to compare.
    :return: True if there is an abbreviated approximate match, False otherwise.
    """
    if '  ' in name1 or '  ' in name2:
        raise ValueError(f'Double spaces in name: {name1}, {name2}')

    if approximate_match(name1, name2) or approximate_match(abbreviate_sci_name(name1), name2) or approximate_match(abbreviate_sci_name(name2),
                                                                                                                    name1):
        return True
    else:
        return False


def precise_match(name1: str, name2: str, allow_any_start_point=None):
    """
    The `precise_match` function checks whether two given names are an exact match.

    :param name1: The first name to compare.
    :param name2: The second name to compare.
    :param allow_any_start_point: Unused. Defaults to None.
    :return: True if the names are an exact match, False otherwise.

    Raises:
        ValueError: If any of the names is not in lower case.

    """
    if '  ' in name1 or '  ' in name2:
        raise ValueError(f'Double spaces in name: {name1}, {name2}')
    if name1.lower() != name1.lower() or name2.lower() != name2.lower():
        raise ValueError(f'Names not lower case: {name1}, {name2}')

    if name1 == name2:
        return True
    else:
        name1_no_spaces = name1.replace(' ', '')
        name2_no_spaces = name2.replace(' ', '')
        if name1_no_spaces == name2_no_spaces:
            return True
        else:
            return False


def approximate_match(name1: str, name2: str, allow_any_start_point: bool = False):
    """
    :param name1: The first name to compare
    :param name2: The second name to compare
    :param allow_any_start_point: [Optional] If set to True, remove '-' as they are sometimes used across line breaks for relationships. And look for any overlap.
     Default is False.
    :return: True if a partial match is found, False otherwise

    This function compares two names and returns True if a partial match is found.
    It checks for possible combinations of words in each name and compares them to find if there is any overlap.

    """
    # When allow_any_start_point is set to true, i.e. for relationships and not scientific names, remove '-' as these are sometimes used across line-
    # breaks

    if precise_match(name1, name2):
        return True
    if allow_any_start_point:
        name1_to_use = name1.replace('-', '')
        name2_to_use = name2.replace('-', '')
    else:
        name1_to_use = name1
        name2_to_use = name2

    name1_split = [word.strip(string.punctuation) for word in name1_to_use.split()]
    name2_split = [word.strip(string.punctuation) for word in name2_to_use.split()]
    if '' in name1_split:
        name1_split.remove('')
    if '' in name2_split:
        name2_split.remove('')

    def get_succesive_combinations_of_words(namelist):
        possible_splits = []
        for i in range(len(namelist)):
            if allow_any_start_point:
                if i > 0:
                    for j in range(i):
                        possible_splits.append(' '.join(namelist[j:i + 1]))
                    possible_splits.append(namelist[i])
                else:
                    possible_splits.append(' '.join(namelist[:i + 1]))
            else:
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


def fuzzy_match(name1: str, name2: str, allow_any_start_point=None):
    """
    The `precise_match` function checks whether two given names are an exact match.

    :param name1: The first name to compare.
    :param name2: The second name to compare.
    :param allow_any_start_point: Unused. Defaults to None.
    :return: True if the names are an exact match, False otherwise.

    Raises:
        ValueError: If any of the names is not in lower case.

    """
    if '  ' in name1 or '  ' in name2:
        raise ValueError(f'Double spaces in name: {name1}, {name2}')
    if name1.lower() != name1.lower() or name2.lower() != name2.lower():
        raise ValueError(f'Names not lower case: {name1}, {name2}')

    ratio = SequenceMatcher(None, name1, name2).ratio()

    if ratio>fuzzy_match_ratio:
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
    # Usually these two things would be the same, but when using an approximate matching method you can end up with unwanted duplications
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
    overlap = set(true_positives_in_ground_truths).intersection(set(false_negatives))
    assert len(ground_truth_names) == len(true_positives_in_ground_truths + false_negatives)
    # print(f'False positives: {false_positives}')
    # print(f'False negatives: {false_negatives}')
    return true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives


def RE_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: str, relationship: str):
    # TODO: check this

    if matching_method == 'precise':
        NER_matching_method = abbreviated_precise_match
        RE_matching_method = precise_match
    elif matching_method == 'approximate':
        NER_matching_method = abbreviated_approximate_match
        RE_matching_method = approximate_match

    elif matching_method == 'fuzzy':
        NER_matching_method = abbreviated_approximate_match
        RE_matching_method = fuzzy_match

    else:
        raise ValueError(f'Unrecognised matching method: {matching_method}')

    true_positives_in_ground_truths = []
    true_positives_in_model_annotations = []
    for model_ann in model_annotations.taxa:
        for model_med_cond in getattr(model_ann, relationship) or []:
            for g in ground_truth_annotations.taxa:
                if NER_matching_method(model_ann.scientific_name, g.scientific_name):
                    for ground_med_condition in getattr(g, relationship) or []:
                        if RE_matching_method(model_med_cond, ground_med_condition, allow_any_start_point=True):
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
                            if RE_matching_method(model_med_cond, ground_med_condition, allow_any_start_point=True):
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
                            if RE_matching_method(m_med_condition, gt_med_cond, allow_any_start_point=True):
                                found = True
                if not found:
                    false_negatives.append(ground_truth_string)
    assert len(false_negatives) == len(set(false_negatives))
    # print(f'False positives: {false_positives}')
    # print(f'False negatives: {false_negatives}')
    return true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives


def check_errors(model_annotations: TaxaData, ground_truth_annotations: TaxaData, out_dir: str, chunk_id: int, model_tag: str):
    ## This gives errors in the relaxed cases.

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
    all = [true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives,
           retrue_positives_in_model_annotations, refalse_positives,
           refalse_negatives, meretrue_positives_in_model_annotations, merefalse_positives, merefalse_negatives]
    padded = list(zip(*itertools.zip_longest(*all, fillvalue='')))
    problems = pd.DataFrame(zip(*padded),
                            columns=['NER_tp_in_ground', 'NER_tp_in_model', 'NER_fp', 'NER_fn', 'MedCond_tp', 'MedCond_fp', 'MedCond_fn', 'MedEff_tp',
                                     'MedEff_fp', 'MedEff_fn'])
    problems.to_csv(os.path.join(out_dir, f'{str(chunk_id)}_{model_tag}_problems.csv'))

def clean_model_annotations_using_taxonomy_knowledge(model_annotations: TaxaData):
    old_taxa_list = []
    for m in model_annotations.taxa:
        old_taxa_list.append(m.scientific_name)
    new_taxa_list = filter_name_list_using_sci_names(old_taxa_list)
    new_anns = []
    for model_ann in model_annotations.taxa:
        if model_ann.scientific_name in new_taxa_list:
            new_anns.append(model_ann)

    return TaxaData(taxa=new_anns)
