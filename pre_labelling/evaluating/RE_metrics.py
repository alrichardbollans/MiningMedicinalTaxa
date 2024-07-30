from typing import Callable

from pre_labelling.evaluating import precise_entity_match, approximate_entity_match, is_annotation_in_annotation_list, get_metrics_from_tp_fp_fn, \
    read_annotation_json


# had to remove from testing.evaluation_methods to make it work

def precise_RE_annotation_match(a1: dict, a2: dict):
    """
    :param a1: dictionary containing the first annotation
    :param a2: dictionary containing the second annotation
    :return: True if the annotations match precisely, False otherwise

    This method compares two annotations to determine if they match precisely. The annotations are considered a match if the following conditions are satisfied:
    - The label of the annotations matches
    - The start and end positions of the 'from_entity' in both annotations are the same
    - The start and end positions of the 'to_entity' in both annotations are the same
    - The text of the 'from_entity' is the same (case-insensitive) in both annotations
    - The text of the 'to_entity' is the same (case-insensitive) in both annotations
    """
    if a1['label'] == a2['label']:
        from_entity1 = a1['from_entity']['value']
        from_entity2 = a2['from_entity']['value']

        to_entity1 = a1['to_entity']['value']
        to_entity2 = a2['to_entity']['value']
        if precise_entity_match(from_entity1, from_entity2) and precise_entity_match(to_entity1, to_entity2):
            return True

    return False


def approximate_RE_annotation_match(a1: dict, a2: dict):
    """
    :param a1: dictionary containing the first annotation
    :param a2: dictionary containing the second annotation
    :return: True if the annotations match precisely, False otherwise

    This method compares two annotations to determine if they match approximately.
    """

    if a1['label'] == a2['label']:
        from_entity1 = a1['from_entity']['value']
        from_entity2 = a2['from_entity']['value']

        to_entity1 = a1['to_entity']['value']
        to_entity2 = a2['to_entity']['value']
        if approximate_entity_match(from_entity1, from_entity2) and approximate_entity_match(to_entity1, to_entity2):
            return True

    return False


def RE_evaluation(model_annotations: list, ground_truth_annotations: list, matching_method: Callable, relationship_class: str = None):
    """

    :param model_annotations: List of model annotations.
    :param ground_truth_annotations: List of ground truth annotations.
    :param matching_method: Method for matching annotations.
    :param relationship_class: Optional. Class of relationship annotations.

    :return: Tuple containing precision, recall, and F1 score.

    """
    if relationship_class is not None:
        model_annotations = [a for a in model_annotations[:] if relationship_class == a['label']]
        ground_truth_annotations = [a for a in ground_truth_annotations[:] if relationship_class == a['label']]
    # true positives
    true_positives = [a for a in model_annotations if is_annotation_in_annotation_list(a, ground_truth_annotations, matching_method)]
    # False positives
    false_positives = [a for a in model_annotations if
                       not is_annotation_in_annotation_list(a, ground_truth_annotations, matching_method)]
    # False negatives
    false_negatives = [a for a in ground_truth_annotations if
                       not is_annotation_in_annotation_list(a, model_annotations, matching_method)]

    return get_metrics_from_tp_fp_fn(true_positives, false_positives, false_negatives)


def example_main():
    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '4187556', '32')
    RE_evaluation(re_annotations, re_annotations, precise_RE_annotation_match)
    RE_evaluation(re_annotations, re_annotations, approximate_RE_annotation_match)
    x, y, z = RE_evaluation(re_annotations, re_annotations, precise_RE_annotation_match, 'has_medicinal_effect')
    assert x == 1
    assert y == 1
    assert z == 1


if __name__ == '__main__':
    example_main()
