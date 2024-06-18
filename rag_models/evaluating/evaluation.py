from typing import List, Callable
import sys

sys.path.append("../testing/evaluation_methods")

from rag_models.gemini import get_response_json
from testing.evaluation_methods import get_metrics_from_tp_fp_fn, is_annotation_in_annotation_list, read_annotation_json


def get_outputs_from_model_annotations(annotations: List[dict]):
    """
    Extracts output information from annotations.

    :param annotations: A list of dictionaries representing annotations.

    :return: A list of dictionaries representing the outputs. Each dictionary contains the following keys -
                - 'from_text': A string representing the text of the from entity value.
                - 'to_text': A string representing the text of the to entity value.
                - 'relationship': A string representing the relationship label.
                - 'from_label': A string representing a label of the from entity.
                - 'to_label': A string representing a label of the to entity.
    """
    outputs = []
    for ann in annotations['annotations']:
        outputs.append({'entity': ann['entity']})
        for med_condition in ann['Medical condition']:
            outputs.append({'entity': ann['entity'], 'Medical condition': med_condition})
        for med_effect in ann['Medicinal effect']:
            outputs.append({'entity': ann['entity'], 'Medicinal effect': med_effect})

    # Get non duplicated outputs
    # TODO:check this
    unique_outputs = [dict(t) for t in {tuple(d.items()) for d in outputs}]
    return unique_outputs


def get_outputs_from_human_annotations(annotations: List[dict]):
    """
    Extracts output information from annotations.

    :param annotations: A list of dictionaries representing annotations.

    :return: A list of dictionaries representing the outputs. Each dictionary contains the following keys -
                - 'from_text': A string representing the text of the from entity value.
                - 'to_text': A string representing the text of the to entity value.
                - 'relationship': A string representing the relationship label.
                - 'from_label': A string representing a label of the from entity.
                - 'to_label': A string representing a label of the to entity.
    """
    outputs = []
    for ann in annotations:
        for from_label in ann['from_entity']['value']['labels']:
            for to_label in ann['to_entity']['value']['labels']:
                outputs.append(
                    {'from_text': ann['from_entity']['value']['text'], 'to_text': ann['to_entity']['value']['text'], 'relationship': ann['label'],
                     'from_label': from_label, 'to_label': to_label})
    # Get non duplicated outputs
    unique_outputs = [dict(t) for t in {tuple(d.items()) for d in outputs}]

    new_outputs = []
    for ann in unique_outputs:
        if ann['from_label'] in TAXON_ENTITY_CLASSES:
            new_outputs.append({'entity': ann['from_text']})
            if ann['relationship'] == 'treats_medical_condition':
                new_outputs.append({'entity': ann['from_text'], 'Medical condition': ann['to_text']})
            if ann['relationship'] == 'has_medicinal_effect':
                new_outputs.append({'entity': ann['from_text'], 'Medicinal effect': ann['to_text']})
        if ann['to_label'] in TAXON_ENTITY_CLASSES:
            new_outputs.append({'entity': ann['to_text']})
            raise ValueError('this should never happen')
    # TODO:check this
    unique_outputs = [dict(t) for t in {tuple(d.items()) for d in new_outputs}]
    return unique_outputs


def precise_output_annotation_match(a1: dict, a2: dict):
    """
    Check if two outputs match exactly i.e. the same entity types, relationship and corresponding text.

    :param a1: The first dictionary.
    :param a2: The second dictionary.
    :return: True if values of corresponding keys are equal in both dictionaries, False otherwise.
    """
    for key in a1.keys():
        if key not in a2.keys():
            return False
        if not a1[key] == a2[key]:
            return False
    for key in a2.keys():
        if key not in a1.keys():
            return False
        if not a1[key] == a2[key]:
            return False
    return True


def approximate_output_annotation_match(a1: dict, a2: dict):
    """
    :param a1: Dictionary containing information about the first annotation.
    :param a2: Dictionary containing information about the second annotation.
    :return: True if the annotations approximately match, False otherwise.

    This method checks whether two annotations approximately match by comparing their attributes. It returns True if the following conditions are met:

    If any of these conditions are not met, the method returns False.
    """
    for key in a1.keys():
        if key not in a2.keys():
            return False
        if (not a1[key].lower() in a2[key].lower()) and (not a2[key].lower() in a1[key].lower()):
            return False

    for key in a2.keys():
        if key not in a1.keys():
            return False
        if (not a1[key].lower() in a2[key].lower()) and (not a2[key].lower() in a1[key].lower()):
            return False

    return True


def chunkwise_evaluation(model_annotations: list, ground_truth_annotations: list, matching_method: Callable):
    """
    Evaluate model performance on a chunkwise basis. Note this can also be extended to papers by including all annotations from a paper.

    :param model_annotations: List of model annotations.
    :param ground_truth_annotations: List of ground truth annotations.
    :param matching_method: Method to determine if an annotation matches another.
    :return: Metrics calculated from true positives, false positives, and false negatives.
    """

    model_outputs = get_outputs_from_model_annotations(model_annotations)
    ground_truth_outputs = get_outputs_from_human_annotations(ground_truth_annotations)

    true_positives = [a for a in model_outputs if is_annotation_in_annotation_list(a, ground_truth_outputs, matching_method)]
    # False positives
    false_positives = [a for a in model_outputs if not is_annotation_in_annotation_list(a, ground_truth_outputs, matching_method)]
    # False negatives
    false_negatives = [a for a in ground_truth_outputs if not is_annotation_in_annotation_list(a, model_outputs, matching_method)]

    return get_metrics_from_tp_fp_fn(true_positives, false_positives, false_negatives)
