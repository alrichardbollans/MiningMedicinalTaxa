import json
import os
import copy

import pandas as pd

ENTITY_CLASSES = ['Scientific Plant Name', 'Scientific Fungus Name', 'Medical Condition', 'Medicinal Effect']

RELATIONS = ['treats_medical_condition', 'has_medicinal_effect']


def pre_evaluation_checks(annotations: dict):
    # Check manual annotations and model annotations for artefacts that we want to avoid
    pass


def read_annotation_json(annotations_directory: str, corpus_id: str, chunk_id: str):
    json_file = os.path.join(annotations_directory, f'task_for_label_studio_{corpus_id}_{chunk_id}.json')
    with open(json_file) as f:
        d = json.load(f)
        print(d)
    results = d[0]['predictions'][0]['result']
    ner_annotations = [c for c in results if c['type'] == 'labels']
    re_annotations = [c for c in results if c['type'] == 'relation']

    # separate annotations by class
    separate_NER_annotations = []
    for ann in ner_annotations:

        for label in ann['value']['labels']:
            new_annotation = copy.deepcopy(ann)
            new_annotation['value']['labels'] = label
            separate_NER_annotations.append(new_annotation)
    pre_evaluation_checks(separate_NER_annotations)
    pre_evaluation_checks(re_annotations)
    return separate_NER_annotations, re_annotations


def precise_NER_annotation_match(a1: dict, a2: dict):
    if a1['value']['end'] == a2['value']['end'] and a1['value']['start'] == a2['value']['start'] and a1['value']['text'].lower() == a2['value'][
        'text'].lower() and a1['value']['labels'] == a2['value']['labels']:

        return True
    else:
        return False


def is_NER_annotation_in_annotations(anntn: dict, ner_annotations: list, matching_method) -> bool:
    for annotation in ner_annotations:
        if matching_method(annotation, anntn):
            return True
    return False


def NER_evaluation(model_annotations, ground_truth_annotations, matching_method, entity_class: str = None):
    if entity_class is not None:
        model_annotations = [a for a in model_annotations[:] if entity_class == a['value']['labels']]
        ground_truth_annotations = [a for a in ground_truth_annotations[:] if entity_class == a['value']['labels']]
    # true positives
    true_positives = [a for a in model_annotations if is_NER_annotation_in_annotations(a, ground_truth_annotations, matching_method)]
    # False positives
    false_positives = [a for a in model_annotations if
                       not is_NER_annotation_in_annotations(a, ground_truth_annotations, matching_method)]
    # False negatives
    false_negatives = [a for a in ground_truth_annotations if
                       not is_NER_annotation_in_annotations(a, model_annotations, matching_method)]
    # precision
    precision = len(true_positives) / len(true_positives + false_positives)
    # recall
    recall = len(true_positives) / len(true_positives + false_negatives)
    # f1
    f1_score = 2 * precision * recall / (precision + recall)
    pass


def main():
    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '4187556', '32')
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match)
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match, 'Medicinal Effect')


if __name__ == '__main__':
    main()
