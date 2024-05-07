import copy
import json
import os
from typing import Callable

import numpy as np

from literature_downloads import get_kword_dict
from testing.evaluation_methods import standardise_NER_annotations, standardise_RE_annotations

ENTITY_CLASSES = ['Scientific Plant Name', 'Scientific Fungus Name', 'Medical Condition', 'Medicinal Effect']

RELATIONS = ['treats_medical_condition', 'has_medicinal_effect']


def read_annotation_json(annotations_directory: str, corpus_id: str, chunk_id: str):
    """
    Reads annotation data from a JSON file.

    :param annotations_directory: The directory where the JSON file is located.
    :param corpus_id: The ID of the corpus.
    :param chunk_id: The ID of the chunk within the corpus.
    :return: A tuple containing two lists: separate_NER_annotations and separate_RE_annotations.
    """
    json_file = os.path.join(annotations_directory, f'task_for_labelstudio_{corpus_id}_chunk_{chunk_id}.json')
    with open(json_file) as f:
        d = json.load(f)
        # print(d)
    results = d[0]['predictions'][0]['result']
    ner_annotations = [c for c in results if c['type'] == 'labels']
    re_annotations = [c for c in results if c['type'] == 'relation']

    ### Resolve entities in RE annotations as don't want to match based on entity number
    for re_annotation in re_annotations:
        from_entity = re_annotation['from_id']
        to_entity = re_annotation['to_id']
        for ner_annotation in ner_annotations:
            if ner_annotation['id'] == from_entity:
                re_annotation['from_entity'] = ner_annotation
            if ner_annotation['id'] == to_entity:
                re_annotation['to_entity'] = ner_annotation

    # separate annotations by class label
    separate_NER_annotations = []
    for ann in ner_annotations:

        for label in ann['value']['labels']:
            new_annotation = copy.deepcopy(ann)
            new_annotation['value']['label'] = label
            del new_annotation['value']['labels']
            separate_NER_annotations.append(new_annotation)

    kw_dict = get_kword_dict()
    standardise_NER_annotations(separate_NER_annotations, kw_dict)

    separate_RE_annotations = []
    for ann in re_annotations:
        for label in ann['labels']:
            new_annotation = copy.deepcopy(ann)
            new_annotation['label'] = label
            del new_annotation['labels']
            separate_RE_annotations.append(new_annotation)

    standardise_RE_annotations(separate_RE_annotations)
    return separate_NER_annotations, separate_RE_annotations


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


def get_metrics_from_tp_fp_fn(true_positives: list, false_positives: list, false_negatives: list):
    """
    Calculate precision, recall, and F1 score given lists of true positives (TP), false positives (FP),
    and false negatives (FN).

    :param true_positives: The number of true positives.
    :param false_positives: The number of false positives.
    :param false_negatives: The number of false negatives.
    :return: A tuple containing the precision, recall, and F1 score.
    """
    # precision
    if len(true_positives + false_positives) != 0:
        precision = len(true_positives) / len(true_positives + false_positives)
    else:
        print(f'Zero Division WARNING: No positive annotations given by model.')
        print(f'Setting precision to np.nan')
        precision = np.nan
    # recall
    if len(true_positives + false_negatives) != 0:
        recall = len(true_positives) / len(true_positives + false_negatives)
    else:
        print(f'Zero Division WARNING: No positive ground truth annotations.')
        print(f'Setting recall to np.nan')
        recall = np.nan
    # f1
    denom = ((2 * len(true_positives)) + len(false_positives) + len(false_negatives))
    if denom != 0:
        f1_score = 2 * len(true_positives) / denom
    else:
        print(f'Zero Division WARNING: No TP,FP,FN.')
        print(f'Setting f1_score to np.nan')
        f1_score = np.nan
    return precision, recall, f1_score


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
