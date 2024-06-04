import json
import pandas as pd
from pathlib import Path
from typing import List, Callable
from testing.evaluation_methods import read_annotation_json, NER_evaluation, precise_NER_annotation_match, approximate_NER_annotation_match
from testing.evaluation_methods import RE_evaluation, precise_RE_annotation_match, approximate_RE_annotation_match
from testing.evaluation_methods import get_metrics_from_tp_fp_fn, get_outputs_from_annotations, precise_output_annotation_match, approximate_output_annotation_match, chunkwise_evaluation
# Function to read JSON file
def read_annotation_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def merge_json_data(*datasets):
    merged_data = []
    for data in datasets:
        merged_data.extend(data)
    return merged_data

def transform_annotation_for_task_id(data, task_id):
    # Debug: Print all items to check for missing 'labels'
    for item in data:
        if "annotations" in item and item["annotations"]:
            for result in item["annotations"][0]["result"]:
                if "type" in result and result["type"] == "relation":
                    if "labels" not in result:
                        print(f"Missing 'labels' in relation: {result}")

    # Filter for the specified task by ID
    task = next((item for item in data if item["id"] == task_id), None)
    if task is None:
        print(f"Task with ID {task_id} not found.")
        return None

    # Transform the structure safely
    transformed_data = [{
        "data": {
            "text": task["data"]["text"]
        },
        "predictions": [{
            "result": [
                *[
                    {
                        "id": result["id"],
                        "from_name": result["from_name"],
                        "to_name": result["to_name"],
                        "type": result["type"],
                        "value": result["value"]
                    } for result in task["annotations"][0]["result"]
                    if result.get("type") == "labels"
                ],
                *[
                    {
                        "from_id": relation["from_id"],
                        "to_id": relation["to_id"],
                        "type": relation["type"],
                        "labels": relation.get("labels", ["Default label"])  # Safe access with default
                    } for relation in task["annotations"][0]["result"]
                    if relation.get("type") == "relation"
                ]
            ]
        }]
    }]

    return transformed_data


def save_transformed_annotation(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)


def calculate_metrics(model_annots, gt_annots, eval_func, match_func, label=None):
    print(f"Calculating metrics for label: {label}")  # Debug output
    model_label_annots = [ann for ann in model_annots if ann.get('label') == label]
    gt_label_annots = [ann for ann in gt_annots if ann.get('label') == label]

    print(f"Model annotations for {label}: {model_label_annots}")  # Debug output
    print(f"GT annotations for {label}: {gt_label_annots}")  # Debug output

    if not model_label_annots and not gt_label_annots:
        return {'Precision': float('nan'), 'Recall': float('nan'), 'F1 Score': float('nan')}
    else:
        precision, recall, f1_score = eval_func(model_label_annots, gt_label_annots, match_func, label)
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }

def calculate_metrics(model_annots, gt_annots, eval_func, match_func, label=None):
    print(f"Calculating metrics for label: {label}")  # Debug output

    # Adjusting the label extraction based on your data structure:
    if label:
        # This assumes that NER labels are nested inside 'value' -> 'labels' and RE labels are directly in 'labels'.
        model_label_annots = [ann for ann in model_annots if (ann.get('value', {}).get('label') == label
                                                              or label in ann.get('labels', [])
                                                              or (ann.get('label') and ann.get('label') == label))]
        gt_label_annots = [ann for ann in gt_annots if (ann.get('value', {}).get('label') == label
                                                        or label in ann.get('labels', [])
                                                        or (ann.get('label') and ann.get('label') == label))]
    else:
        # When no label is specified, all annotations should be processed (likely for overall RE/NER metrics).
        model_label_annots = model_annots
        gt_label_annots = gt_annots

    print(f"Model annotations for {label}: {model_label_annots}")  # Debug output
    print(f"GT annotations for {label}: {gt_label_annots}")  # Debug output

    if not model_label_annots and not gt_label_annots:
        return {'Precision': float('nan'), 'Recall': float('nan'), 'F1 Score': float('nan')}
    else:
        precision, recall, f1_score = eval_func(model_label_annots, gt_label_annots, match_func, label)
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }


def extract_ids(filename):
    parts = filename.replace('.json', '').split('_')
    specific_id = parts[3]
    chunk_id = parts[-1]
    return specific_id, chunk_id


def process_annotations(manual_filenames, tasks_completed_dir, manual_transformed_dir):
    results = []
    ner_labels = ['Scientific Plant Name', 'Scientific Fungus Name', 'Medical Condition', 'Medicinal Effect']
    re_labels = ['treats_medical_condition', 'has_medicinal_effect']

    for filename in manual_filenames:
        specific_id, chunk_id = extract_ids(filename)
        try:
            model_NER_annotations, model_RE_annotations = read_annotation_json(tasks_completed_dir, specific_id, chunk_id)
            gt_NER_annotations, gt_RE_annotations = read_annotation_json(manual_transformed_dir, specific_id, chunk_id)

            result = {'Chunk_id': chunk_id, 'Specific_id': specific_id}

            for precision_type, ner_match_func, re_match_func in [
                ('Precise', precise_NER_annotation_match, precise_RE_annotation_match),
                ('Approx', approximate_NER_annotation_match, approximate_RE_annotation_match)]:

                # Calculate NER metrics for all labels
                ner_all_metrics = calculate_metrics(model_NER_annotations, gt_NER_annotations, NER_evaluation, ner_match_func)
                prefix = f'NER_{precision_type}_all_'
                result.update({prefix + k: v for k, v in ner_all_metrics.items()})

                # Calculate RE metrics for all labels
                re_all_metrics = calculate_metrics(model_RE_annotations, gt_RE_annotations, RE_evaluation, re_match_func)
                prefix = f'RE_{precision_type}_all_'
                result.update({prefix + k: v for k, v in re_all_metrics.items()})

                # Calculate metrics for specific NER labels
                for ner_label in ner_labels:
                    ner_label_metrics = calculate_metrics(model_NER_annotations, gt_NER_annotations, NER_evaluation, ner_match_func, ner_label)
                    prefix = f'NER_{precision_type}_{ner_label.replace(" ", "_")}_'
                    result.update({prefix + k: v for k, v in ner_label_metrics.items()})

                # Calculate metrics for specific RE labels
                for re_label in re_labels:
                    re_label_metrics = calculate_metrics(model_RE_annotations, gt_RE_annotations, RE_evaluation, re_match_func, re_label)
                    prefix = f'RE_{precision_type}_{re_label.replace(" ", "_")}_'
                    result.update({prefix + k: v for k, v in re_label_metrics.items()})

            results.append(result)
        except KeyError as e:
            print(f"KeyError: {e} for Specific ID: {specific_id}, Chunk ID: {chunk_id}")
        except Exception as e:
            print(f"Unexpected error: {e} for Specific ID: {specific_id}, Chunk ID: {chunk_id}")

    return pd.DataFrame(results)


def process_annotations_chunkwise(manual_filenames, tasks_completed_dir, manual_transformed_dir):
    results = []
    ner_labels = ['Scientific Plant Name', 'Scientific Fungus Name', 'Medical Condition', 'Medicinal Effect']
    re_labels = ['treats_medical_condition', 'has_medicinal_effect']

    for filename in manual_filenames:
        specific_id, chunk_id = extract_ids(filename)  # Assuming extract_ids function to get identifiers from filename

        # Read model and ground truth annotations for the given chunk
        model_annotations, gt_annotations = read_annotation_json(tasks_completed_dir, specific_id, chunk_id)
        gt_transformed_annotations = read_annotation_json(manual_transformed_dir, specific_id, chunk_id)

        # Store chunk-specific results
        result = {'Chunk_id': chunk_id, 'Specific_id': specific_id}

        # Define evaluation types and corresponding matching methods
        for precision_type, ner_match_func, re_match_func in [
            ('Precise', precise_output_annotation_match, precise_output_annotation_match),
            ('Approx', approximate_output_annotation_match, approximate_output_annotation_match)]:

            # Calculate metrics for NER
            ner_metrics = chunkwise_evaluation(model_annotations, gt_transformed_annotations, ner_match_func)
            print(f"NER Metrics for {precision_type}: {ner_metrics}, type: {type(ner_metrics)}")  # Debugging line

            if isinstance(ner_metrics, tuple) and len(ner_metrics) == 3:  # Check if the result is a tuple of length 3
                prefix = f'NER_{precision_type}_'
                result.update({
                    prefix + 'Precision': ner_metrics[0],
                    prefix + 'Recall': ner_metrics[1],
                    prefix + 'F1_Score': ner_metrics[2]
                })
            else:
                print(f"Error: NER metrics for {precision_type} are not as expected.")

            # Calculate metrics for RE
            re_metrics = chunkwise_evaluation(model_annotations, gt_transformed_annotations, re_match_func)
            print(f"RE Metrics for {precision_type}: {re_metrics}, type: {type(re_metrics)}")  # Debugging line

            if isinstance(re_metrics, tuple) and len(re_metrics) == 3:  # Check if the result is a tuple of length 3
                prefix = f'RE_{precision_type}_'
                result.update({
                    prefix + 'Precision': re_metrics[0],
                    prefix + 'Recall': re_metrics[1],
                    prefix + 'F1_Score': re_metrics[2]
                })
            else:
                print(f"Error: RE metrics for {precision_type} are not as expected.")

            # Calculate metrics for specific NER and RE labels
            for ner_label in ner_labels:
                ner_label_metrics = chunkwise_evaluation(model_annotations, gt_transformed_annotations, ner_match_func)
                print(
                    f"NER Label Metrics for {precision_type} - {ner_label}: {ner_label_metrics}, type: {type(ner_label_metrics)}")  # Debugging line

                if isinstance(ner_label_metrics, tuple) and len(
                        ner_label_metrics) == 3:  # Check if the result is a tuple of length 3
                    prefix = f'NER_{precision_type}_{ner_label.replace(" ", "_")}_'
                    result.update({
                        prefix + 'Precision': ner_label_metrics[0],
                        prefix + 'Recall': ner_label_metrics[1],
                        prefix + 'F1_Score': ner_label_metrics[2]
                    })
                else:
                    print(f"Error: NER label metrics for {precision_type} - {ner_label} are not as expected.")

            for re_label in re_labels:
                re_label_metrics = chunkwise_evaluation(model_annotations, gt_transformed_annotations, re_match_func)
                print(
                    f"RE Label Metrics for {precision_type} - {re_label}: {re_label_metrics}, type: {type(re_label_metrics)}")  # Debugging line

                if isinstance(re_label_metrics, tuple) and len(
                        re_label_metrics) == 3:  # Check if the result is a tuple of length 3
                    prefix = f'RE_{precision_type}_{re_label.replace(" ", "_")}_'
                    result.update({
                        prefix + 'Precision': re_label_metrics[0],
                        prefix + 'Recall': re_label_metrics[1],
                        prefix + 'F1_Score': re_label_metrics[2]
                    })
                else:
                    print(f"Error: RE label metrics for {precision_type} - {re_label} are not as expected.")

        results.append(result)

    return results


def save_df_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False)  # index=False to avoid saving the index column
        print(f"DataFrame saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save DataFrame to CSV: {e}")

def calculate_summary_metrics(df):
    # Calculate the mean and standard deviation, excluding NaN values
    mean_metrics = df.drop(columns=['Chunk_id', 'Specific_id']).apply(pd.to_numeric, errors='coerce').mean(skipna=True).to_frame(name='Mean').T
    std_metrics = df.drop(columns=['Chunk_id', 'Specific_id']).apply(pd.to_numeric, errors='coerce').std(skipna=True).to_frame(name='Std').T

    # Concatenate the mean and standard deviation DataFrames
    summary_metrics = pd.concat([mean_metrics, std_metrics])

    # Transpose the DataFrame and add metric names as index
    summary_metrics = summary_metrics.transpose().rename_axis('Metrics').reset_index()

    return summary_metrics

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
        if 'from_entity' in ann and 'to_entity' in ann:
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

def test_ner_re():
    # Load JSON data for NER and RE annotations
    ner_annotations, re_annotations = read_annotation_json('testing/test_medicinal_01/tasks_completed', '228197190', '6')
    gt_ner_annotations, gt_re_annotations = read_annotation_json('testing/test_medicinal_01/manual_annotation_transformed',
                                                                 '228197190', '6')

    # Test precise NER match
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match)
    print(f"Precise NER - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise NER match Scientific Plant Name
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                                 'Scientific Plant Name')
    print(f"Precise NER Scientific Plant Name - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise NER match Scientific Fungus Name
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                                 'Scientific Fungus Name')
    print(f"Precise NER Scientific Fungus Name - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise NER match Medical Condition
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                                 'Medical Condition')
    print(f"Precise NER Medical Condition - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise NER match Medicinal Effect
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                                 'Medicinal Effect')
    print(f"Precise NER Medicinal Effect - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate NER match
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match)
    print(f"Approximate NER - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate NER match Scientific Plant Name
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                                 'Scientific Plant Name')
    print(f"Approximate NER Scientific Plant Name - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate NER match Scientific Fungus Name
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                                 'Scientific Fungus Name')
    print(f"Approximate NER Scientific Fungus Name - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate NER match Medical Condition
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                                 'Medical Condition')
    print(f"Approximate NER Medical Condition - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate NER match Medicinal Effect
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                                 'Medicinal Effect')
    print(f"Approximate NER Medicinal Effect - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise RE match
    precision, recall, f1_score = RE_evaluation(re_annotations, gt_re_annotations, precise_RE_annotation_match)
    print(f"Precise RE - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise RE match treats_medical_condition
    precision, recall, f1_score = RE_evaluation(re_annotations, gt_re_annotations, precise_RE_annotation_match,
                                                'treats_medical_condition')
    print(f"Precise RE treats_medical_condition - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test precise RE match has_medicinal_effects
    precision, recall, f1_score = RE_evaluation(re_annotations, gt_re_annotations, precise_RE_annotation_match,
                                                'has_medicinal_effects')
    print(f"Precise RE has_medicinal_effects - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate RE match treats_medical_condition
    precision, recall, f1_score = RE_evaluation(re_annotations, gt_re_annotations, approximate_RE_annotation_match,
                                                'treats_medical_condition')
    print(f"Approximate RE treats_medical_condition - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Test approximate RE match has_medicinal_effects
    precision, recall, f1_score = RE_evaluation(re_annotations, gt_re_annotations, approximate_RE_annotation_match,
                                                'has_medicinal_effects')
    print(f"Approximate RE has_medicinal_effects - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")




# The main function for loading and processing
def main():
    data_filenames = [#'testing/test_medicinal_01/manual_annotation/task_for_labelstudio_80818116_ifra_fc.json',
                      #'testing/test_medicinal_01/manual_annotation/task_for_labelstudio_4187756.json',
                      'testing/test_medicinal_01/manual_annotation/task_for_labelstudio_161880242_228197190_268329601_4187556_360558516_80818116.json'
                      ]
    data = merge_json_data(*[load_json_data(f) for f in data_filenames])

    task_id_to_output = {

        1: 'task_for_labelstudio_161880242_chunk_0.json',
        2: 'task_for_labelstudio_161880242_chunk_2.json',
        3: 'task_for_labelstudio_161880242_chunk_3.json',
        4: 'task_for_labelstudio_161880242_chunk_4.json',
        5: 'task_for_labelstudio_161880242_chunk_5.json',
        6: 'task_for_labelstudio_161880242_chunk_11.json',
        7: 'task_for_labelstudio_161880242_chunk_12.json',
        8: 'task_for_labelstudio_161880242_chunk_25.json',
        9: 'task_for_labelstudio_161880242_chunk_26.json',
        #10: 'task_for_labelstudio_161880242_chunk_27.json', # excluded has only ref
        #11: 'task_for_labelstudio_161880242_chunk_30.json', # excluded has only ref
        #12: 'task_for_labelstudio_161880242_chunk_31.json', # excluded has only ref
        #13: 'task_for_labelstudio_161880242_chunk_32.json', # excluded has only ref
        #14: 'task_for_labelstudio_161880242_chunk_24.json', # wrong parsing
        #15: 'task_for_labelstudio_161880242_chunk_1.json', # wrong parsing # excluded as table of content
        #16: 'task_for_labelstudio_161880242_chunk_29.json', # excluded has only ref
        #17: 'task_for_labelstudio_228197190_chunk_0.json', # wrong parsing
        18: 'task_for_labelstudio_228197190_chunk_2.json',
        19: 'task_for_labelstudio_228197190_chunk_3.json',
        #20: 'task_for_labelstudio_228197190_chunk_4.json', wrong parsing
        21: 'task_for_labelstudio_228197190_chunk_5.json',
        22: 'task_for_labelstudio_228197190_chunk_6.json',
        23: 'task_for_labelstudio_228197190_chunk_7.json',
        24: 'task_for_labelstudio_228197190_chunk_8.json',
        25: 'task_for_labelstudio_228197190_chunk_9.json',
        26: 'task_for_labelstudio_228197190_chunk_10.json',
        #27: 'task_for_labelstudio_228197190_chunk_14.json', # wrong parsing
        #28: 'task_for_labelstudio_228197190_chunk_15.json', # wrong parsing
        29: 'task_for_labelstudio_228197190_chunk_16.json',
        30: 'task_for_labelstudio_228197190_chunk_17.json',
        31: 'task_for_labelstudio_228197190_chunk_18.json',
        #32: 'task_for_labelstudio_228197190_chunk_19.json', # wrong parsing
        33: 'task_for_labelstudio_228197190_chunk_20.json',
        34: 'task_for_labelstudio_228197190_chunk_21.json',
        #35: 'task_for_labelstudio_228197190_chunk_26.json', # wrong parsing
        36: 'task_for_labelstudio_228197190_chunk_28.json',
        37: 'task_for_labelstudio_228197190_chunk_30.json',
        #38: 'task_for_labelstudio_228197190_chunk_31.json', # excluded has only ref
        #39: 'task_for_labelstudio_228197190_chunk_32.json', # excluded has only ref
        #40: 'task_for_labelstudio_228197190_chunk_33.json', # excluded has only ref
        #41: 'task_for_labelstudio_228197190_chunk_34.json', # excluded has only ref
        42: 'task_for_labelstudio_228197190_chunk_35.json',
        43: 'task_for_labelstudio_228197190_chunk_36.json',
        44: 'task_for_labelstudio_228197190_chunk_37.json',
        #45: 'task_for_labelstudio_228197190_chunk_38.json', # wrong parsing
        #46: 'task_for_labelstudio_228197190_chunk_39.json' # wrong parsing
        47: 'task_for_labelstudio_197989253_chunk_1.json',
        #48: 'task_for_labelstudio_197989253_chunk_3.json' # wrong parsing # excluded has only ref
        #49: 'task_for_labelstudio_197989253_chunk_4.json',# excluded has only ref
        50: 'task_for_labelstudio_197989253_chunk_29.json',
        51: 'task_for_labelstudio_197989253_chunk_31.json',
        #52: 'task_for_labelstudio_197989253_chunk_45.json'
        53: 'task_for_labelstudio_197989253_chunk_46.json',
        54: 'task_for_labelstudio_197989253_chunk_47.json',
        55: 'task_for_labelstudio_197989253_chunk_48.json',
        56: 'task_for_labelstudio_197989253_chunk_49.json',
        57: 'task_for_labelstudio_197989253_chunk_50.json',
        #58: 'task_for_labelstudio_197989253_chunk_56.json', # wrong parsing
        ### TO UPDATE WHEN T FINISHES
        #59: 'task_for_labelstudio_268329601_chunk_2.json', # wrong parsing
        #60: 'task_for_labelstudio_268329601_chunk_3.json', # wrong parsing
        #61: 'task_for_labelstudio_268329601_chunk_4.json', # wrong parsing
        #62: 'task_for_labelstudio_268329601_chunk_6.json', # wrong parsing
        #63: 'task_for_labelstudio_268329601_chunk_9.json', # wrong parsing
        64: 'task_for_labelstudio_268329601_chunk_12.json',
        65: 'task_for_labelstudio_268329601_chunk_13.json',
        #66: 'task_for_labelstudio_268329601_chunk_14.json', # wrong parsing
        67: 'task_for_labelstudio_268329601_chunk_15.json',
        #68: 'task_for_labelstudio_268329601_chunk_16.json', # wrong parsing
        69: 'task_for_labelstudio_268329601_chunk_18.json',
        70: 'task_for_labelstudio_268329601_chunk_19.json',
        71: 'task_for_labelstudio_268329601_chunk_20.json',
        #72: 'task_for_labelstudio_268329601_chunk_21.json', # wrong parsing
        73: 'task_for_labelstudio_268329601_chunk_22.json',
        #74: 'task_for_labelstudio_268329601_chunk_23.json', # wrong parsing
        75: 'task_for_labelstudio_268329601_chunk_24.json',
        76: 'task_for_labelstudio_268329601_chunk_25.json',
        #77: 'task_for_labelstudio_268329601_chunk_26.json', # wrong parsing
        78: 'task_for_labelstudio_268329601_chunk_27.json',
        #79: 'task_for_labelstudio_268329601_chunk_28.json', # wrong parsing
        #80: 'task_for_labelstudio_268329601_chunk_29.json', # wrong parsing
        81: 'task_for_labelstudio_268329601_chunk_30.json',
        82: 'task_for_labelstudio_268329601_chunk_31.json',
        #83: 'task_for_labelstudio_268329601_chunk_32.json', # wrong parsing
        84: 'task_for_labelstudio_268329601_chunk_33.json',
        85: 'task_for_labelstudio_268329601_chunk_34.json',
        #86: 'task_for_labelstudio_268329601_chunk_35.json', # wrong parsing
        87: 'task_for_labelstudio_268329601_chunk_36.json',
        #88: 'task_for_labelstudio_268329601_chunk_37.json', # wrong parsing
        #89: 'task_for_labelstudio_268329601_chunk_38.json', # wrong parsing
        90: 'task_for_labelstudio_268329601_chunk_39.json',
        #91: 'task_for_labelstudio_268329601_chunk_40.json', # wrong parsing
        #92: 'task_for_labelstudio_268329601_chunk_41.json', # wrong parsing
        93: 'task_for_labelstudio_268329601_chunk_42.json',
        #94: 'task_for_labelstudio_268329601_chunk_43.json', # wrong parsing
        #95: 'task_for_labelstudio_268329601_chunk_44.json', # wrong parsing
        96: 'task_for_labelstudio_268329601_chunk_45.json',
        97: 'task_for_labelstudio_268329601_chunk_46.json',
        98: 'task_for_labelstudio_268329601_chunk_48.json',
        99: 'task_for_labelstudio_268329601_chunk_49.json',
        100: 'task_for_labelstudio_268329601_chunk_50.json',
        101: 'task_for_labelstudio_268329601_chunk_51.json',
        102: 'task_for_labelstudio_268329601_chunk_52.json',
        103: 'task_for_labelstudio_268329601_chunk_53.json',
        104: 'task_for_labelstudio_268329601_chunk_54.json',
        105: 'task_for_labelstudio_268329601_chunk_55.json',
        106: 'task_for_labelstudio_268329601_chunk_56.json',
        107: 'task_for_labelstudio_268329601_chunk_57.json',
        108: 'task_for_labelstudio_268329601_chunk_58.json',
        109: 'task_for_labelstudio_268329601_chunk_59.json',
        110: 'task_for_labelstudio_268329601_chunk_60.json',
        111: 'task_for_labelstudio_268329601_chunk_62.json',
        112: 'task_for_labelstudio_268329601_chunk_63.json',
        113: 'task_for_labelstudio_268329601_chunk_64.json',
        #114: 'task_for_labelstudio_268329601_chunk_65.json', #wrong parsing
        #115:'task_for_labelstudio_268329601_chunk_66.json', # wrong parsing
        #116:'task_for_labelstudio_268329601_chunk_68.json', # wrong parsing
        117: 'task_for_labelstudio_268329601_chunk_70.json',
        118: 'task_for_labelstudio_268329601_chunk_72.json',
        119: 'task_for_labelstudio_268329601_chunk_73.json',
        120: 'task_for_labelstudio_268329601_chunk_74.json',
        121: 'task_for_labelstudio_268329601_chunk_75.json',
        122: 'task_for_labelstudio_268329601_chunk_76.json',
        #123: 'task_for_labelstudio_268329601_chunk_86.json', # excluded has only ref
        #124: 'task_for_labelstudio_268329601_chunk_87.json', # wrong parsing  # excluded has only ref
        #125: 'task_for_labelstudio_268329601_chunk_88.json', # wrong parsing  # excluded has only ref
        #126: 'task_for_labelstudio_268329601_chunk_89.json', # wrong parsing  # excluded has only ref
        #127: 'task_for_labelstudio_268329601_chunk_90.json', # wrong parsing  # excluded has only ref
        #128: 'task_for_labelstudio_268329601_chunk_91.json', # wrong parsing  # excluded has only ref
        #129: 'task_for_labelstudio_268329601_chunk_92.json', # excluded has only ref
        #130: 'task_for_labelstudio_268329601_chunk_94.json', # wrong parsing  # excluded has only ref
        #131: 'task_for_labelstudio_268329601_chunk_95.json', # wrong parsing  # excluded has only ref
        #132: 'task_for_labelstudio_268329601_chunk_96.json', # wrong parsing  # excluded has only ref
        #133: 'task_for_labelstudio_268329601_chunk_97.json',  # excluded has only ref
        #134: 'task_for_labelstudio_268329601_chunk_98.json', # wrong parsing  # excluded has only ref
        #135: 'task_for_labelstudio_268329601_chunk_99.json', # wrong parsing # excluded has only ref
        #136: 'task_for_labelstudio_268329601_chunk_107.json', # wrong parsing # excluded has only ref
        #137: 'task_for_labelstudio_268329601_chunk_108.json', # excluded has only ref
        #138: 'task_for_labelstudio_268329601_chunk_109.json', # excluded has only ref
        #139: 'task_for_labelstudio_268329601_chunk_110.json', # excluded has only ref
        #140: 'task_for_labelstudio_268329601_chunk_111.json', # excluded has only ref
        #141: 'task_for_labelstudio_268329601_chunk_112.json', # excluded has only ref
        #142: 'task_for_labelstudio_268329601_chunk_114.json', # wrong parsing # excluded has only ref
        #143: 'task_for_labelstudio_268329601_chunk_115.json',  # excluded has only ref
        #144: 'task_for_labelstudio_268329601_chunk_116.json',  # excluded has only ref
        #145: 'task_for_labelstudio_268329601_chunk_7.json', # wrong parsing
        #146: 'task_for_labelstudio_268329601_chunk_8.json', # wrong parsing
        #147: 'task_for_labelstudio_268329601_chunk_47.json', # wrong parsing
        #148: 'task_for_labelstudio_268329601_chunk_67.json', # wrong parsing
        #149: 'task_for_labelstudio_268329601_chunk_69.json', # wrong parsing
        # added by me
        #150: 'task_for_labelstudio_268329601_chunk_100.json', # excluded as only ref
        #151: 'task_for_labelstudio_268329601_chunk_101.json', # excluded as only ref
        #152: 'task_for_labelstudio_268329601_chunk_102.json', # excluded as only ref
        #153: 'task_for_labelstudio_268329601_chunk_106.json', # excluded as only ref
        154: 'task_for_labelstudio_4187756_chunk_0.json',
        155: 'task_for_labelstudio_4187756_chunk_15.json',
        156: 'task_for_labelstudio_4187756_chunk_19.json',
        157: 'task_for_labelstudio_4187756_chunk_22.json',
        158: 'task_for_labelstudio_4187756_chunk_31.json',
        #159: 'task_for_labelstudio_4187756_chunk_32.json', # excluded has only ref
        # Adam chunks
        160: 'task_for_labelstudio_360558516_chunk_4.json',
        161: 'task_for_labelstudio_360558516_chunk_7.json',
        162: 'task_for_labelstudio_360558516_chunk_8.json',
        163: 'task_for_labelstudio_360558516_chunk_9.json',
        164: 'task_for_labelstudio_360558516_chunk_19.json',
        # ifra
        #165: 'task_for_labelstudio_80818116_chunk_26.json', # wrong parsing
        166: 'task_for_labelstudio_80818116_chunk_28.json',
        #167: 'task_for_labelstudio_80818116_chunk_29.json', # wrong parsing
        168: 'task_for_labelstudio_80818116_chunk_30.json',
        169: 'task_for_labelstudio_80818116_chunk_111.json',
        170: 'task_for_labelstudio_80818116_chunk_36.json',
        #171: 'task_for_labelstudio_80818116_chunk_54.json', # wrong parsing
        172: 'task_for_labelstudio_80818116_chunk_55.json',
        173: 'task_for_labelstudio_80818116_chunk_56.json',
        174: 'task_for_labelstudio_80818116_chunk_57.json',
        175: 'task_for_labelstudio_80818116_chunk_59.json',
        176: 'task_for_labelstudio_80818116_chunk_63.json',
        #177: 'task_for_labelstudio_80818116_chunk_66.json', # wrong parsing
        #178: 'task_for_labelstudio_80818116_chunk_77.json', # wrong parsing
        179: 'task_for_labelstudio_80818116_chunk_76.json',
        180: 'task_for_labelstudio_80818116_chunk_75.json',
       # 181: 'task_for_labelstudio_80818116_chunk_72.json',
       # 182: 'task_for_labelstudio_80818116_chunk_71.json',
       # 183: 'task_for_labelstudio_80818116_chunk_69.json',
       # 184: 'task_for_labelstudio_80818116_chunk_68.json',
       # 185: 'task_for_labelstudio_80818116_chunk_67.json',
       # 186: 'task_for_labelstudio_80818116_chunk_78.json',
       # 187: 'task_for_labelstudio_80818116_chunk_167.json'
    }  # Example task IDs

    for task_id, output_filename in task_id_to_output.items():
        transformed_result = transform_annotation_for_task_id(data, task_id)
        if transformed_result:
            output_path = f'testing/test_medicinal_01/manual_annotation_transformed/{output_filename}'
            save_transformed_annotation(transformed_result, output_path)
            print(f"Saved transformed data for task ID {task_id} to {output_path}")

    manual_filenames = [
        'task_for_labelstudio_161880242_chunk_0.json',
        'task_for_labelstudio_161880242_chunk_2.json',
        'task_for_labelstudio_161880242_chunk_3.json',
        'task_for_labelstudio_161880242_chunk_4.json',
        'task_for_labelstudio_161880242_chunk_5.json',
        'task_for_labelstudio_161880242_chunk_11.json',
        'task_for_labelstudio_161880242_chunk_12.json',
        'task_for_labelstudio_161880242_chunk_25.json',
        'task_for_labelstudio_161880242_chunk_26.json',
        # 'task_for_labelstudio_161880242_chunk_27.json', # excluded has only ref
        # 'task_for_labelstudio_161880242_chunk_30.json', # excluded has only ref
        # 'task_for_labelstudio_161880242_chunk_31.json', # excluded has only ref
        # 'task_for_labelstudio_161880242_chunk_32.json', # excluded has only ref
        # 'task_for_labelstudio_161880242_chunk_24.json', # wrong parsing
        # 'task_for_labelstudio_161880242_chunk_1.json', # wrong parsing # excluded as table of content
        # 'task_for_labelstudio_161880242_chunk_29.json', # excluded has only ref
        # 'task_for_labelstudio_228197190_chunk_0.json', wrong parsing
        'task_for_labelstudio_228197190_chunk_2.json',
        'task_for_labelstudio_228197190_chunk_3.json',
        # 'task_for_labelstudio_228197190_chunk_4.json', wrong parsing
        'task_for_labelstudio_228197190_chunk_5.json',
        'task_for_labelstudio_228197190_chunk_6.json',
        'task_for_labelstudio_228197190_chunk_7.json',
        'task_for_labelstudio_228197190_chunk_8.json',
        'task_for_labelstudio_228197190_chunk_9.json',
        'task_for_labelstudio_228197190_chunk_10.json',
        # 'task_for_labelstudio_228197190_chunk_14.json', # wrong parsing
        # 'task_for_labelstudio_228197190_chunk_15.json', # wrong parsing
        'task_for_labelstudio_228197190_chunk_16.json',
        'task_for_labelstudio_228197190_chunk_17.json',
        'task_for_labelstudio_228197190_chunk_18.json',
        # 'task_for_labelstudio_228197190_chunk_19.json', # wrong parsing
        'task_for_labelstudio_228197190_chunk_20.json',
        'task_for_labelstudio_228197190_chunk_21.json',
        # 'task_for_labelstudio_228197190_chunk_26.json', # wrong parsing
        'task_for_labelstudio_228197190_chunk_28.json',
        'task_for_labelstudio_228197190_chunk_30.json',
        # 'task_for_labelstudio_228197190_chunk_31.json', # excluded has only ref
        # 'task_for_labelstudio_228197190_chunk_32.json', # excluded has only ref
        # 'task_for_labelstudio_228197190_chunk_33.json', # excluded has only ref
        # 'task_for_labelstudio_228197190_chunk_34.json', # excluded has only ref
        'task_for_labelstudio_228197190_chunk_35.json',
        'task_for_labelstudio_228197190_chunk_36.json',
        'task_for_labelstudio_228197190_chunk_37.json',
        # 'task_for_labelstudio_228197190_chunk_38.json', # wrong parsing
        # 'task_for_labelstudio_228197190_chunk_39.json' # wrong parsing
        'task_for_labelstudio_197989253_chunk_1.json',
        # 'task_for_labelstudio_197989253_chunk_3.json' # wrong parsing # excluded has only ref
        # 'task_for_labelstudio_197989253_chunk_4.json',# excluded has only ref
        'task_for_labelstudio_197989253_chunk_29.json',
        'task_for_labelstudio_197989253_chunk_31.json',
        # 'task_for_labelstudio_197989253_chunk_45.json'
        'task_for_labelstudio_197989253_chunk_46.json',
        'task_for_labelstudio_197989253_chunk_47.json',
        'task_for_labelstudio_197989253_chunk_48.json',
        'task_for_labelstudio_197989253_chunk_49.json',
        'task_for_labelstudio_197989253_chunk_50.json',
        # 'task_for_labelstudio_197989253_chunk_56.json', # wrong parsing
        ### TO UPDATE WHEN T FINISHES
        # 'task_for_labelstudio_268329601_chunk_2.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_3.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_4.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_6.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_9.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_12.json',
        'task_for_labelstudio_268329601_chunk_13.json',
        # 'task_for_labelstudio_268329601_chunk_14.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_15.json',
        # 'task_for_labelstudio_268329601_chunk_16.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_18.json',
        'task_for_labelstudio_268329601_chunk_19.json',
        'task_for_labelstudio_268329601_chunk_20.json',
        # 'task_for_labelstudio_268329601_chunk_21.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_22.json',
        # 'task_for_labelstudio_268329601_chunk_23.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_24.json',
        'task_for_labelstudio_268329601_chunk_25.json',
        # 'task_for_labelstudio_268329601_chunk_26.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_27.json',
        # 'task_for_labelstudio_268329601_chunk_28.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_29.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_30.json',
        'task_for_labelstudio_268329601_chunk_31.json',
        # 'task_for_labelstudio_268329601_chunk_32.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_33.json',
        'task_for_labelstudio_268329601_chunk_34.json',
        # 'task_for_labelstudio_268329601_chunk_35.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_36.json',
        # 'task_for_labelstudio_268329601_chunk_37.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_38.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_39.json',
        # 'task_for_labelstudio_268329601_chunk_40.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_41.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_42.json',
        # 'task_for_labelstudio_268329601_chunk_43.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_44.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_45.json',
        'task_for_labelstudio_268329601_chunk_46.json',
        'task_for_labelstudio_268329601_chunk_48.json',
        'task_for_labelstudio_268329601_chunk_49.json',
        'task_for_labelstudio_268329601_chunk_50.json',
        'task_for_labelstudio_268329601_chunk_51.json',
        'task_for_labelstudio_268329601_chunk_52.json',
        'task_for_labelstudio_268329601_chunk_53.json',
        'task_for_labelstudio_268329601_chunk_54.json',
        'task_for_labelstudio_268329601_chunk_55.json',
        'task_for_labelstudio_268329601_chunk_56.json',
        'task_for_labelstudio_268329601_chunk_57.json',
        'task_for_labelstudio_268329601_chunk_58.json',
        'task_for_labelstudio_268329601_chunk_59.json',
        'task_for_labelstudio_268329601_chunk_60.json',
        'task_for_labelstudio_268329601_chunk_62.json',
        'task_for_labelstudio_268329601_chunk_63.json',
        'task_for_labelstudio_268329601_chunk_64.json',
        # 'task_for_labelstudio_268329601_chunk_65.json', #wrong parsing
        # 'task_for_labelstudio_268329601_chunk_66.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_68.json', # wrong parsing
        'task_for_labelstudio_268329601_chunk_70.json',
        'task_for_labelstudio_268329601_chunk_72.json',
        'task_for_labelstudio_268329601_chunk_73.json',
        'task_for_labelstudio_268329601_chunk_74.json',
        'task_for_labelstudio_268329601_chunk_75.json',
        'task_for_labelstudio_268329601_chunk_76.json',
        # 'task_for_labelstudio_268329601_chunk_86.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_87.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_88.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_89.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_90.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_91.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_92.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_94.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_95.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_96.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_97.json',  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_98.json', # wrong parsing  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_99.json', # wrong parsing # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_107.json', # wrong parsing # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_108.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_109.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_110.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_111.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_112.json', # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_114.json', # wrong parsing # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_115.json',  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_116.json',  # excluded has only ref
        # 'task_for_labelstudio_268329601_chunk_7.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_8.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_47.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_67.json', # wrong parsing
        # 'task_for_labelstudio_268329601_chunk_69.json', # wrong parsing
        # added by me
        # 'task_for_labelstudio_268329601_chunk_100.json', # excluded as only ref
        # 'task_for_labelstudio_268329601_chunk_101.json', # excluded as only ref
        # 'task_for_labelstudio_268329601_chunk_102.json', # excluded as only ref
        # 'task_for_labelstudio_268329601_chunk_106.json', # excluded as only ref
        'task_for_labelstudio_4187756_chunk_0.json',
        'task_for_labelstudio_4187756_chunk_15.json',
        'task_for_labelstudio_4187756_chunk_19.json',
        'task_for_labelstudio_4187756_chunk_22.json',
        'task_for_labelstudio_4187756_chunk_31.json',
        # 'task_for_labelstudio_4187756_chunk_32.json', # excluded has only ref
        # Adam chunks
        'task_for_labelstudio_360558516_chunk_4.json',
        'task_for_labelstudio_360558516_chunk_7.json',
        'task_for_labelstudio_360558516_chunk_8.json',
        'task_for_labelstudio_360558516_chunk_9.json',
        'task_for_labelstudio_360558516_chunk_19.json',
        # ifra
        # 'task_for_labelstudio_80818116_chunk_26.json', # wrong parsing
        'task_for_labelstudio_80818116_chunk_28.json',
        # 'task_for_labelstudio_80818116_chunk_29.json', # wrong parsing
        'task_for_labelstudio_80818116_chunk_30.json',
        'task_for_labelstudio_80818116_chunk_111.json',
        'task_for_labelstudio_80818116_chunk_36.json',
        # 'task_for_labelstudio_80818116_chunk_54.json', # wrong parsing
        'task_for_labelstudio_80818116_chunk_55.json',
        'task_for_labelstudio_80818116_chunk_56.json',
        'task_for_labelstudio_80818116_chunk_57.json',
        'task_for_labelstudio_80818116_chunk_59.json',
        'task_for_labelstudio_80818116_chunk_63.json',
        # 'task_for_labelstudio_80818116_chunk_66.json', wrong parsing
        # 'task_for_labelstudio_80818116_chunk_77.json', wrong parsing
        'task_for_labelstudio_80818116_chunk_76.json',
        'task_for_labelstudio_80818116_chunk_75.json'
        'task_for_labelstudio_80818116_chunk_76.json',
        'task_for_labelstudio_80818116_chunk_75.json',
        #'task_for_labelstudio_80818116_chunk_72.json', wrong parsing
        #'task_for_labelstudio_80818116_chunk_71.json',
        #'task_for_labelstudio_80818116_chunk_69.json',
        #'task_for_labelstudio_80818116_chunk_68.json', on hold
        #'task_for_labelstudio_80818116_chunk_67.json',
        #'task_for_labelstudio_80818116_chunk_78.json',
        #'task_for_labelstudio_80818116_chunk_167.json'

    ]  # Example filenames
    tasks_completed_dir = 'testing/test_medicinal_01/tasks_completed'
    manual_transformed_dir = 'testing/test_medicinal_01/manual_annotation_transformed'
    df = process_annotations(manual_filenames, tasks_completed_dir, manual_transformed_dir)
    print(df.head())

    save_df_to_csv(df, 'testing/test_medicinal_01/all_metrics.csv')
    summary_metrics = calculate_summary_metrics(df)
    save_df_to_csv(summary_metrics, 'testing/test_medicinal_01/summary_metrics.csv')

    # Process annotations with chunkwise evaluation
    df_chunkwise = process_annotations_chunkwise(manual_filenames, tasks_completed_dir, manual_transformed_dir)
    print(df_chunkwise.head())

    # Save chunkwise evaluation results
    save_df_to_csv(df_chunkwise, 'testing/test_medicinal_01/chunkwise_metrics.csv')

    # Calculate and save summary metrics
    summary_metrics_chunkwise = calculate_summary_metrics(df_chunkwise)
    save_df_to_csv(summary_metrics_chunkwise, 'testing/test_medicinal_01/summary_chunkwise_metrics.csv')
    # Call the testing function
    test_ner_re()


if __name__ == '__main__':
    main()

