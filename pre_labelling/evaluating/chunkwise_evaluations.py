from typing import List, Callable

from pre_labelling.evaluating import is_annotation_in_annotation_list, get_metrics_from_tp_fp_fn, read_annotation_json


def get_outputs_from_annotations(annotations: List[dict]):
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
        # Add debug print statements to check the structure of each annotation
        if 'from_entity' not in ann:
            print(f"Missing 'from_entity' in annotation: {ann}")
            continue
        if 'to_entity' not in ann:
            print(f"Missing 'to_entity' in annotation: {ann}")
            continue
        if 'value' not in ann['from_entity']:
            print(f"Missing 'value' in 'from_entity': {ann['from_entity']}")
            continue
        if 'value' not in ann['to_entity']:
            print(f"Missing 'value' in 'to_entity': {ann['to_entity']}")
            continue
        if 'labels' not in ann['from_entity']['value']:
            print(f"Missing 'labels' in 'from_entity.value': {ann['from_entity']['value']}")
            continue
        if 'labels' not in ann['to_entity']['value']:
            print(f"Missing 'labels' in 'to_entity.value': {ann['to_entity']['value']}")
            continue

        for from_label in ann['from_entity']['value']['labels']:
            for to_label in ann['to_entity']['value']['labels']:
                outputs.append(
                    {'from_text': ann['from_entity']['value']['text'], 'to_text': ann['to_entity']['value']['text'], 'relationship': ann['label'],
                     'from_label': from_label, 'to_label': to_label})
    # Get non duplicated outputs
    unique_outputs = [dict(t) for t in {tuple(d.items()) for d in outputs}]
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
    - The 'from_label' attribute of a1 and a2 are equal.
    - The 'to_label' attribute of a1 and a2 are equal.
    - The 'relationship' attribute of a1 and a2 are equal.
    - The from_text of a1 is contained in a2 or vice versa.
    - The to_text of a1 is contained in a2 or vice versa.

    If any of these conditions are not met, the method returns False.
    """
    if a1['from_label'] == a2['from_label'] and a1['to_label'] == a2['to_label'] and a1['relationship'] == a2['relationship']:
        if a1['from_text'].lower() in a2['from_text'].lower() or a2['from_text'].lower() in a1['from_text'].lower():
            if a1['to_text'].lower() in a2['to_text'].lower() or a2['to_text'].lower() in a1['to_text'].lower():
                return True

    return False


def chunkwise_evaluation(model_annotations: list, ground_truth_annotations: list, matching_method: Callable):
    """
    Evaluate model performance on a chunkwise basis. Note this can also be extended to papers by including all annotations from a paper.

    :param model_annotations: List of model annotations.
    :param ground_truth_annotations: List of ground truth annotations.
    :param matching_method: Method to determine if an annotation matches another.
    :return: Metrics calculated from true positives, false positives, and false negatives.
    """
    model_outputs = get_outputs_from_annotations(model_annotations)
    ground_truth_outputs = get_outputs_from_annotations(ground_truth_annotations)

    true_positives = [a for a in model_outputs if is_annotation_in_annotation_list(a, ground_truth_outputs, matching_method)]
    # False positives
    false_positives = [a for a in model_outputs if not is_annotation_in_annotation_list(a, ground_truth_outputs, matching_method)]
    # False negatives
    false_negatives = [a for a in ground_truth_outputs if not is_annotation_in_annotation_list(a, model_outputs, matching_method)]

    return get_metrics_from_tp_fp_fn(true_positives, false_positives, false_negatives)


def example_main():
    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '4187556', '32')
    ner_annotations2, re_annotations2 = read_annotation_json('../test_medicinal_01/tasks_completed', '35321774', '57')
    precision, recall, f1_score = chunkwise_evaluation(re_annotations, re_annotations2, precise_output_annotation_match)
    assert precision == 0
    assert recall == 0
    assert f1_score == 0
    precision, recall, f1_score = chunkwise_evaluation(re_annotations, re_annotations2, approximate_output_annotation_match)
    assert precision == 0
    assert recall == 0
    assert f1_score == 0


if __name__ == '__main__':
    example_main()
