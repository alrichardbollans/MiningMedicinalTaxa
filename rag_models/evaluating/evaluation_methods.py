from typing import List, Callable
import sys

from rag_models.structured_output_schema import TaxaData

sys.path.append("../testing/evaluation_methods")

from testing.evaluation_methods import get_metrics_from_tp_fp_fn


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
    if name1 in name2 or name2 in name1:
        return True
    else:
        return False


def NER_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: Callable):
    """

    """

    ground_truth_names = [g.scientific_name for g in ground_truth_annotations.taxa]
    assert len(ground_truth_names) == len(ground_truth_annotations.taxa)
    assert len(ground_truth_names) == len(set(ground_truth_names))

    true_positives = []
    # TODO: fix this for recall, in approximate case could potentially be repetitions so break isn't working
    for a in model_annotations.taxa:
        for g in ground_truth_annotations.taxa:
            if matching_method(a.scientific_name, g.scientific_name):
                true_positives.append(a.scientific_name)
                break
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

    return get_metrics_from_tp_fp_fn(true_positives, false_positives, false_negatives)


def RE_evaluation(model_annotations: TaxaData, ground_truth_annotations: TaxaData, matching_method: Callable, relationship: str):
    # TODO: finish this
    pass


def check_errors(model_annotations: TaxaData, ground_truth_annotations: TaxaData):
    pass
