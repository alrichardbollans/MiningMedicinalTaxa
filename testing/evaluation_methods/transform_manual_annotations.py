import json
import pandas as pd
from pathlib import Path
from NER_metrics import read_annotation_json, NER_evaluation, precise_NER_annotation_match, approximate_NER_annotation_match
from RE_metrics import RE_evaluation, precise_RE_annotation_match, approximate_RE_annotation_match

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
        model_NER_annotations, model_RE_annotations = read_annotation_json(tasks_completed_dir, specific_id, chunk_id)
        gt_NER_annotations, gt_RE_annotations = read_annotation_json(manual_transformed_dir, specific_id, chunk_id)

        result = {'Chunk_id': chunk_id, 'Specific_id': specific_id}

        for precision_type, ner_match_func, re_match_func in [
            ('Precise', precise_NER_annotation_match, precise_RE_annotation_match),
            ('Approx', approximate_NER_annotation_match, approximate_RE_annotation_match)]:

            # Calculate NER metrics for all labels
            ner_all_metrics = calculate_metrics(model_NER_annotations, gt_NER_annotations, NER_evaluation,
                                                ner_match_func)
            prefix = f'NER_{precision_type}_all_'
            result.update({prefix + k: v for k, v in ner_all_metrics.items()})

            # Calculate RE metrics for all labels (possibly missing this?)
            re_all_metrics = calculate_metrics(model_RE_annotations, gt_RE_annotations, RE_evaluation,
                                               re_match_func)
            prefix = f'RE_{precision_type}_all_'
            result.update({prefix + k: v for k, v in re_all_metrics.items()})

            # Calculate metrics for specific NER labels
            for ner_label in ner_labels:
                ner_label_metrics = calculate_metrics(model_NER_annotations, gt_NER_annotations, NER_evaluation,
                                                      ner_match_func, ner_label)
                prefix = f'NER_{precision_type}_{ner_label.replace(" ", "_")}_'
                result.update({prefix + k: v for k, v in ner_label_metrics.items()})

            # Calculate metrics for specific RE labels
            for re_label in re_labels:
                re_label_metrics = calculate_metrics(model_RE_annotations, gt_RE_annotations, RE_evaluation,
                                                     re_match_func, re_label)
                prefix = f'RE_{precision_type}_{re_label.replace(" ", "_")}_'
                result.update({prefix + k: v for k, v in re_label_metrics.items()})

        results.append(result)
    return pd.DataFrame(results)


def save_df_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False)  # index=False to avoid saving the index column
        print(f"DataFrame saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save DataFrame to CSV: {e}")


def test_ner_re():
    # Load JSON data for NER and RE annotations
    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '228197190', '6')
    gt_ner_annotations, gt_re_annotations = read_annotation_json('../test_medicinal_01/manual_annotation_transformed',
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

def calculate_summary_metrics(df):
    # Calculate the mean and standard deviation, excluding NaN values
    mean_metrics = df.drop(columns=['Chunk_id', 'Specific_id']).apply(pd.to_numeric, errors='coerce').mean(skipna=True).to_frame(name='Mean').T
    std_metrics = df.drop(columns=['Chunk_id', 'Specific_id']).apply(pd.to_numeric, errors='coerce').std(skipna=True).to_frame(name='Std').T

    # Concatenate the mean and standard deviation DataFrames
    summary_metrics = pd.concat([mean_metrics, std_metrics])

    # Transpose the DataFrame and add metric names as index
    summary_metrics = summary_metrics.transpose().rename_axis('Metrics').reset_index()

    return summary_metrics




# The main function for loading and processing
def main():
    data_filenames = ['../test_medicinal_01/manual_annotation/task_for_labelstudio_80818116_ifra_FC_09_04_2024.json',
                      '../test_medicinal_01/manual_annotation/task_for_labelstudio_4187756.json',
                      '../test_medicinal_01/manual_annotation/task_for_labelstudio_161880242_228197190.json'
                      ]
    data = merge_json_data(*[load_json_data(f) for f in data_filenames])

    task_id_to_output = {
        # 278: 'task_for_labelstudio_80818116_chunk_26.json', # wrong parsing
        279: 'task_for_labelstudio_80818116_chunk_28.json',
        # 280: 'task_for_labelstudio_80818116_chunk_29.json', # wrong parsing
        281: 'task_for_labelstudio_80818116_chunk_30.json',
        282: 'task_for_labelstudio_80818116_chunk_111.json',
        433: 'task_for_labelstudio_4187756_chunk_22.json',
        434: 'task_for_labelstudio_4187756_chunk_31.json',
        # 435: 'task_for_labelstudio_4187756_chunk_32.json', # excluded has only ref
        1: 'task_for_labelstudio_161880242_chunk_0.json',
        2: 'task_for_labelstudio_161880242_chunk_2.json',
        3: 'task_for_labelstudio_161880242_chunk_3.json',
        4: 'task_for_labelstudio_161880242_chunk_4.json',
        5: 'task_for_labelstudio_161880242_chunk_5.json',
        6: 'task_for_labelstudio_161880242_chunk_11.json',
        7: 'task_for_labelstudio_161880242_chunk_12.json',
        8: 'task_for_labelstudio_161880242_chunk_25.json',
        9: 'task_for_labelstudio_161880242_chunk_26.json',
        # 10: 'task_for_labelstudio_161880242_chunk_27.json', # excluded has only ref
        # 11: 'task_for_labelstudio_161880242_chunk_30.json', # excluded has only ref
        # 12: 'task_for_labelstudio_161880242_chunk_31.json', # excluded has only ref
        # 13: 'task_for_labelstudio_161880242_chunk_32.json', # excluded has only ref
        # : 'task_for_labelstudio_161880242_chunk_1.json', # wrong parsing
        # : 'task_for_labelstudio_161880242_chunk_24.json', # wrong parsing
        # : 'task_for_labelstudio_161880242_chunk_29.json', # wrong parsing
        14: 'task_for_labelstudio_228197190_chunk_0.json',
        15: 'task_for_labelstudio_228197190_chunk_2.json',
        16: 'task_for_labelstudio_228197190_chunk_3.json',
        # 17: 'task_for_labelstudio_228197190_chunk_4.json', wrong parsing
        18: 'task_for_labelstudio_228197190_chunk_5.json',
        19: 'task_for_labelstudio_228197190_chunk_6.json',
        20: 'task_for_labelstudio_228197190_chunk_7.json',
        21: 'task_for_labelstudio_228197190_chunk_8.json',
        22: 'task_for_labelstudio_228197190_chunk_9.json',
        23: 'task_for_labelstudio_228197190_chunk_10.json',
        # 24: 'task_for_labelstudio_228197190_chunk_14.json', # wrong parsing
        # 25: 'task_for_labelstudio_228197190_chunk_15.json', # wrong parsing
        26: 'task_for_labelstudio_228197190_chunk_16.json',
        27: 'task_for_labelstudio_228197190_chunk_17.json',
        28: 'task_for_labelstudio_228197190_chunk_18.json',
        # 29: 'task_for_labelstudio_228197190_chunk_19.json', # wrong parsing
        30: 'task_for_labelstudio_228197190_chunk_20.json',
        31: 'task_for_labelstudio_228197190_chunk_21.json',
        # 32: 'task_for_labelstudio_228197190_chunk_26.json', # wrong parsing
        33: 'task_for_labelstudio_228197190_chunk_28.json',
        34: 'task_for_labelstudio_228197190_chunk_30.json',
        # 35: 'task_for_labelstudio_228197190_chunk_31.json', # excluded has only ref
        # 36: 'task_for_labelstudio_228197190_chunk_32.json', # excluded has only ref
        # 37: 'task_for_labelstudio_228197190_chunk_33.json', # excluded has only ref
        # 38: 'task_for_labelstudio_228197190_chunk_34.json', # excluded has only ref
        39: 'task_for_labelstudio_228197190_chunk_35.json',
        40: 'task_for_labelstudio_228197190_chunk_36.json',
        41: 'task_for_labelstudio_228197190_chunk_37.json',
        # 42: 'task_for_labelstudio_228197190_chunk_38.json', # wrong parsing
        # 43: 'task_for_labelstudio_228197190_chunk_39.json' # wrong parsing
    }  # Example task IDs

    for task_id, output_filename in task_id_to_output.items():
        transformed_result = transform_annotation_for_task_id(data, task_id)
        if transformed_result:
            output_path = f'../test_medicinal_01/manual_annotation_transformed/{output_filename}'
            save_transformed_annotation(transformed_result, output_path)
            print(f"Saved transformed data for task ID {task_id} to {output_path}")

    manual_filenames = [
        # 'task_for_labelstudio_80818116_chunk_26.json', wrong parsing
        'task_for_labelstudio_80818116_chunk_28.json',
        # 'task_for_labelstudio_80818116_chunk_29.json', wrong parsing
        'task_for_labelstudio_80818116_chunk_30.json'
        'task_for_labelstudio_80818116_chunk_111.json',
        'task_for_labelstudio_4187756_chunk_0.json',
        'task_for_labelstudio_4187756_chunk_15.json',
        'task_for_labelstudio_4187756_chunk_19.json',
        'task_for_labelstudio_4187756_chunk_22.json',
        'task_for_labelstudio_4187756_chunk_31.json',
        'task_for_labelstudio_161880242_chunk_0.json',
        'task_for_labelstudio_161880242_chunk_2.json',
        'task_for_labelstudio_161880242_chunk_3.json',
        'task_for_labelstudio_161880242_chunk_4.json',
        'task_for_labelstudio_161880242_chunk_5.json',
        'task_for_labelstudio_161880242_chunk_11.json',
        'task_for_labelstudio_161880242_chunk_12.json',
        # 'task_for_labelstudio_161880242_chunk_24.json', wrong parsing
        'task_for_labelstudio_161880242_chunk_25.json',
        'task_for_labelstudio_161880242_chunk_26.json',
        'task_for_labelstudio_161880242_chunk_27.json'
        'task_for_labelstudio_228197190_chunk_0.json',
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
        'task_for_labelstudio_228197190_chunk_35.json',
        'task_for_labelstudio_228197190_chunk_36.json',
        'task_for_labelstudio_228197190_chunk_37.json',
        # 'task_for_labelstudio_228197190_chunk_38.json', # wrong parsing
        # 'task_for_labelstudio_228197190_chunk_39.json' # wrong parsing
    ]  # Example filenames
    tasks_completed_dir = '../test_medicinal_01/tasks_completed'
    manual_transformed_dir = '../test_medicinal_01/manual_annotation_transformed'
    df = process_annotations(manual_filenames, tasks_completed_dir, manual_transformed_dir)
    print(df.head())

    save_df_to_csv(df, '../test_medicinal_01/all_metrics.csv')
    summary_metrics = calculate_summary_metrics(df)
    save_df_to_csv(summary_metrics, '../test_medicinal_01/summary_metrics.csv')
    # Call the testing function
    test_ner_re()


if __name__ == '__main__':
    main()


