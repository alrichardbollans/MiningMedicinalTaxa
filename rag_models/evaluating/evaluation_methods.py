import string
from typing import Callable

from rag_models.structured_output_schema import TaxaData


def precise_match(name1: str, name2: str):
    """

    """
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
            if i > 0:
                for j in range(i):
                    possible_splits.append(' '.join(namelist[j:i + 1]))
                possible_splits.append(namelist[i])
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


def RE_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: Callable, relationship: str):
    # TODO: check this

    true_positives_in_ground_truths = []
    true_positives_in_model_annotations = []
    for model_ann in model_annotations.taxa:
        for model_med_cond in getattr(model_ann, relationship) or []:
            for g in ground_truth_annotations.taxa:
                if matching_method(model_ann.scientific_name, g.scientific_name):
                    for ground_med_condition in getattr(g, relationship) or []:
                        if matching_method(model_med_cond, ground_med_condition):
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
            if not (any(matching_method(model_ann.scientific_name, g.scientific_name) for g in ground_truth_annotations.taxa)):
                # If no matching scientific name then this is a false positive
                false_positives.append(model_string)
            else:
                found = False
                for g in ground_truth_annotations.taxa:
                    if matching_method(model_ann.scientific_name, g.scientific_name):
                        for ground_med_condition in getattr(g, relationship) or []:
                            if matching_method(model_med_cond, ground_med_condition):
                                found = True
                if not found:
                    false_positives.append(model_string)
    assert len(false_positives) == len(set(false_positives))

    # False negatives
    false_negatives = []
    for g in ground_truth_annotations.taxa:
        for gt_med_cond in getattr(g, relationship) or []:
            ground_truth_string = '_'.join([g.scientific_name, gt_med_cond])
            if not (any(matching_method(g.scientific_name, a.scientific_name) for a in model_annotations.taxa)):
                false_negatives.append(ground_truth_string)
            else:
                found = False
                for m in model_annotations.taxa:
                    if matching_method(m.scientific_name, g.scientific_name):
                        for m_med_condition in getattr(m, relationship) or []:
                            if matching_method(m_med_condition, gt_med_cond):
                                found = True
                if not found:
                    false_negatives.append(ground_truth_string)
    assert len(false_negatives) == len(set(false_negatives))
    print(f'False positives: {false_positives}')
    print(f'False negatives: {false_negatives}')
    return true_positives_in_ground_truths, true_positives_in_model_annotations, false_positives, false_negatives


def check_errors(model_annotations: TaxaData, ground_truth_annotations: TaxaData):
    # TODO: output errors in model_annotations
    pass
