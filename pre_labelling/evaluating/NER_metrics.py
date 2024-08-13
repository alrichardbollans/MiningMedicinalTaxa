import os
from typing import Callable

import numpy as np


def precise_entity_match(entity1: dict, entity2: dict):
    """
    :param entity1: A dictionary representing the first entity to be matched. It should have keys 'start', 'end', and 'text'.
    :param entity2: A dictionary representing the second entity to be matched. It should have keys 'start', 'end', and 'text'.
    :return: A boolean value indicating whether the two entities are a precise match.
    """
    if entity1['start'] == entity2['start'] and entity1['end'] == entity2['end'] and entity1['text'].lower() == entity2['text'].lower():

        return True
    else:
        return False


def precise_NER_annotation_match(a1: dict, a2: dict):
    """
    This method compares two annotations to determine if they match precisely. The annotations are considered a match if the following conditions are satisfied:
    - The label of the annotations matches
    - The start and end positions of the annotations are the same
    - The text is the same (case-insensitive) in both annotations
    """
    entity1 = a1['value']
    entity2 = a2['value']

    if precise_entity_match(entity1, entity2) and entity1['label'] == entity2['label']:

        return True
    else:
        return False


def approximate_entity_match(entity1: dict, entity2: dict):
    """
    Check if entity1 and entity2 are approximate matches. i.e. if one entity subsumes another.

    :param entity1: A dictionary representing the first entity with keys 'start', 'end', and 'text'.
    :param entity2: A dictionary representing the second entity with keys 'start', 'end', and 'text'.
    :return: True if entity1 and entity2 are approximate matches, False otherwise.
    """
    if entity1['start'] <= entity2['start'] and entity1['end'] >= entity2['end'] and entity2['text'].lower() in entity1['text'].lower():
        # If entity1 contains entity2
        return True
    elif entity2['start'] <= entity1['start'] and entity2['end'] >= entity1['end'] and entity1['text'].lower() in entity2['text'].lower():
        # If entity2 contains entity1
        return True
    else:
        return False


def approximate_NER_annotation_match(a1: dict, a2: dict):
    """
    :param a1: A dictionary representing the first entity with 'value' and 'label' keys.
    :param a2: A dictionary representing the second entity with 'value' and 'label' keys.
    :return: A boolean indicating whether the first entity approximately matches the second entity.

    This method checks if one entity is a substring of the other and if their labels are the same. It returns True if the conditions are met, otherwise False.
    """
    # Matches if one is a substring of the other.
    # From: Subramaniam, L. Venkata, et al. "Information extraction from biomedical literature: methodology, evaluation and an application."
    # Proceedings of the twelfth international conference on Information and knowledge management. 2003.
    # Used in: Tsai, Richard Tzong-Han, et al. "Various criteria in the evaluation of biomedical named entity recognition." BMC bioinformatics 7 (2006): 1-8.
    # and
    # Le Guillarme, Nicolas, and Wilfried Thuiller. "TaxoNERD: deep neural models for the recognition of taxonomic entities in the ecological and evolutionary literature."
    # Methods in Ecology and Evolution 13.3 (2022): 625-641.
    entity1 = a1['value']
    entity2 = a2['value']

    if approximate_entity_match(entity1, entity2) and entity1['label'] == entity2['label']:

        return True
    else:
        return False
    pass


def is_annotation_in_annotation_list(model_annotation: dict, annotation_list: list, matching_method) -> bool:
    """
    Check if a given annotation is present in the list of annotations.

    :param model_annotation: The annotation to check.
    :param annotation_list: The list of annotations to search in.
    :param matching_method: The method to check if two annotations match.
    :return: True if the annotation is found, False otherwise.
    :rtype: bool
    """
    for annotation in annotation_list:
        if matching_method(annotation, model_annotation):
            return True
    return False


def NER_evaluation(model_annotations: list, ground_truth_annotations: list, matching_method: Callable, entity_class: str = None):
    """
    Calculate the precision, recall, and F1 score for Named Entity Recognition (NER) evaluation.

    :param model_annotations: List of model-generated annotations.
    :param ground_truth_annotations: List of ground truth annotations.
    :param matching_method: Method used for matching annotations.
    :param entity_class: Optional parameter to filter annotations by entity class.
    :return: A tuple containing the precision, recall, and F1 score.
    """
    if entity_class is not None:
        model_annotations = [a for a in model_annotations[:] if entity_class == a['value']['label']]
        ground_truth_annotations = [a for a in ground_truth_annotations[:] if entity_class == a['value']['label']]
    # true positives
    true_positives = [a for a in model_annotations if is_annotation_in_annotation_list(a, ground_truth_annotations, matching_method)]
    # False positives
    false_positives = [a for a in model_annotations if
                       not is_annotation_in_annotation_list(a, ground_truth_annotations, matching_method)]
    # False negatives
    false_negatives = [a for a in ground_truth_annotations if
                       not is_annotation_in_annotation_list(a, model_annotations, matching_method)]

    precision, recall, f1 = get_metrics_from_tp_fp_fn(true_positives, false_positives, false_negatives)
    return precision, recall, f1


def example_main():
    ner_annotations, re_annotations = read_annotation_json(os.path.join('..', 'test_medicinal_01', 'tasks_completed'), '4187556', '32')
    ner_annotations2, re_annotations2 = read_annotation_json(os.path.join('..', 'test_medicinal_01', 'tasks_completed'), '4187756', '0')
    NER_evaluation(ner_annotations + ner_annotations2, ner_annotations, approximate_NER_annotation_match)
    NER_evaluation(ner_annotations, ner_annotations + ner_annotations2, approximate_NER_annotation_match, 'Medicinal Effect')
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match)
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match, 'Medicinal Effect')


if __name__ == '__main__':
    example_main()
