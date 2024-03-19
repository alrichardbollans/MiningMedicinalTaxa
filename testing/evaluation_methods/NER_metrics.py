import json
import os
import copy

ENTITY_CLASSES = ['Scientific Plant Name', 'Scientific Fungus Name', 'Medical Condition', 'Medicinal Effect']

RELATIONS = ['treats_medical_condition', 'has_medicinal_effect']


# TODO: Set up pre evaluation checks to check annotations for things like leading/trailing whitespace
# TODO: Write some unittests

def pre_evaluation_checks(annotations: list):
    # Check manual annotations and model annotations for artefacts that we want to avoid
    pass


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
        print(d)
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

    pre_evaluation_checks(separate_NER_annotations)

    separate_RE_annotations = []
    for ann in re_annotations:
        for label in ann['labels']:
            new_annotation = copy.deepcopy(ann)
            new_annotation['label'] = label
            del new_annotation['labels']
            separate_RE_annotations.append(new_annotation)

    pre_evaluation_checks(separate_RE_annotations)
    return separate_NER_annotations, separate_RE_annotations


def precise_NER_annotation_match(a1: dict, a2: dict):
    """
    This method compares two annotations to determine if they match precisely. The annotations are considered a match if the following conditions are satisfied:
    - The label of the annotations matches
    - The start and end positions of the annotations are the same
    - The text is the same (case-insensitive) in both annotations
    """
    entity1 = a1['value']
    entity2 = a2['value']

    if entity1['end'] == entity2['end'] and entity1['start'] == entity2['start'] and entity1['text'].lower() == entity2['text'].lower() and entity1[
        'label'] == entity2['label']:

        return True
    else:
        return False


def relaxed_NER_annotation_match(a1: dict, a2: dict):
    pass


def is_annotation_in_annotation_list(model_annotation: dict, ner_annotations: list, matching_method) -> bool:
    """
    Check if a given annotation is present in the list of annotations.

    :param model_annotation: The annotation to check.
    :param ner_annotations: The list of annotations to search in.
    :param matching_method: The method to check if two annotations match.
    :return: True if the annotation is found, False otherwise.
    :rtype: bool
    """
    for annotation in ner_annotations:
        if matching_method(annotation, model_annotation):
            return True
    return False


def NER_evaluation(model_annotations, ground_truth_annotations, matching_method, entity_class: str = None):
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
    # precision
    precision = len(true_positives) / len(true_positives + false_positives)
    # recall
    recall = len(true_positives) / len(true_positives + false_negatives)
    # f1
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def example_main():
    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '4187556', '32')
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match)
    NER_evaluation(ner_annotations, ner_annotations, precise_NER_annotation_match, 'Medicinal Effect')


if __name__ == '__main__':
    example_main()
