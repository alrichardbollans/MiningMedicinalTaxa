import json
from NER_metrics import read_annotation_json, NER_evaluation, precise_NER_annotation_match, approximate_NER_annotation_match
from RE_metrics import RE_evaluation, precise_RE_annotation_match, approximate_RE_annotation_match

def read_annotation_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def merge_json_data(data1, data2):
    return data1 +data2

def transform_annotation_for_task_id(data, task_id):
    """
    Transform the structure of a specific task identified by `task_id` within the provided `data`.

    Parameters:
    - data: The full dataset as a list of dictionaries.
    - task_id: The specific ID of the task to filter for and transform.

    Returns:
    - A transformed data structure for the specific task.
    - None if the task with the specified ID is not found.
    """
    # Filter for the specified task by ID
    task = next((item for item in data if item["id"] == task_id), None)

    # Check if the task exists
    if task is None:
        print(f"Task with ID {task_id} not found.")
        return None

    # Transform the structure
    transformed_data = [
        {
            "data": {
                "text": task["data"]["text"]
            },
            "predictions": [
                {
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
                                "labels": relation["labels"]
                            } for relation in task["annotations"][0]["result"]
                            if relation.get("type") == "relation"
                        ]
                    ]
                }
            ]
        }
    ]

    # Return the transformed structure
    return transformed_data

def save_transformed_annotation(data, filename):
    """
    Save the transformed data to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def main():
    data_filename1 = '../test_medicinal_01/manual_annotation/task_for_labelstudio_80818116_ifra_FC_09_04_2024.json'
    data_filename2 = '../test_medicinal_01/manual_annotation/task_for_labelstudio_4187756.json'
    data_filename3 = '../test_medicinal_01/manual_annotation/task_for_labelstudio_161880242.json'

    data1 = load_json_data(data_filename1)
    data2 = load_json_data(data_filename2)
    data3 = load_json_data(data_filename3)

    data = merge_json_data(data1, data2)
    data = merge_json_data(data,data3)

    task_id_to_output = {
        278: 'task_for_labelstudio_80818116_chunk_26.json',
        279: 'task_for_labelstudio_80818116_chunk_28.json',
        280: 'task_for_labelstudio_80818116_chunk_29.json',
        281: 'task_for_labelstudio_80818116_chunk_30.json',
        282: 'task_for_labelstudio_80818116_chunk_111.json',
        272: 'task_for_labelstudio_4187756_chunk_0.json',
        273: 'task_for_labelstudio_4187756_chunk_15.json',
        274: 'task_for_labelstudio_4187756_chunk_19.json',
        275: 'task_for_labelstudio_4187756_chunk_22.json',
        276: 'task_for_labelstudio_4187756_chunk_31.json',
        1: 'task_for_labelstudio_161880242_chunk_0.json',
        2: 'task_for_labelstudio_161880242_chunk_2.json',
        3: 'task_for_labelstudio_161880242_chunk_3.json',
        4: 'task_for_labelstudio_161880242_chunk_4.json',
        #5: 'task_for_labelstudio_161880242_chunk_5.json',
        6: 'task_for_labelstudio_161880242_chunk_11.json',
        7: 'task_for_labelstudio_161880242_chunk_12.json',
        8: 'task_for_labelstudio_161880242_chunk_25.json',
        9: 'task_for_labelstudio_161880242_chunk_26.json',
        10: 'task_for_labelstudio_161880242_chunk_27.json'
    }

    for task_id, output_filename in task_id_to_output.items():
        transformed_result = transform_annotation_for_task_id(data, task_id)
        if transformed_result:
            output_path = f'../test_medicinal_01/manual_annotation_transformed/{output_filename}'
            save_transformed_annotation(transformed_result, output_path)
            print(f"Saved transformed data for task ID {task_id} to {output_path}")

    # Example filenames from manual_annotation_transformed
    manual_filenames = [
        'task_for_labelstudio_80818116_chunk_26.json',
        'task_for_labelstudio_80818116_chunk_28.json',
        'task_for_labelstudio_80818116_chunk_29.json',
        'task_for_labelstudio_80818116_chunk_30.json',
        'task_for_labelstudio_80818116_chunk_111.json',
        'task_for_labelstudio_4187756_chunk_0.json',
        'task_for_labelstudio_4187756_chunk_15.json',
        'task_for_labelstudio_4187756_chunk_19.json',
        'task_for_labelstudio_4187756_chunk_22.json',
        'task_for_labelstudio_4187756_chunk_31.json'
        'task_for_labelstudio_161880242_chunk_0.json',
        'task_for_labelstudio_161880242_chunk_2.json',
        'task_for_labelstudio_161880242_chunk_3.json',
        'task_for_labelstudio_161880242_chunk_4.json',
       # 'task_for_labelstudio_161880242_chunk_5.json',
        'task_for_labelstudio_161880242_chunk_11.json',
        'task_for_labelstudio_161880242_chunk_12.json',
        'task_for_labelstudio_161880242_chunk_25.json',
        'task_for_labelstudio_161880242_chunk_26.json',
        'task_for_labelstudio_161880242_chunk_27.json'
    ]

    # Model NER and Ground Truth NER annotations lists
    model_NER_annotations = []
    gt_NER_annotations = []
    model_RE_annotations = []
    gt_RE_annotations = []


    # Base directory paths
    tasks_completed_dir = '../test_medicinal_01/tasks_completed'
    manual_transformed_dir = '../test_medicinal_01/manual_annotation_transformed'

    # Function to extract the specific ID and chunk ID from the filename
    def extract_ids(filename):
        parts = filename.replace('.json', '').split('_')
        specific_id = parts[3]  # for example '80818116' is the 4th element after split by '_'
        chunk_id = parts[-1]  # chunk number is the last element after split
        # Debug print to confirm values
        print("Extracting IDs from:", filename, "-> specific_id:", specific_id, ", chunk_id:", chunk_id)
        return specific_id, chunk_id

    # Loop through the manual annotation filenames
    for filename in manual_filenames:
        # Extract specific ID and chunk ID from the filename
        specific_id, chunk_id = extract_ids(filename)

        # Read model-generated annotations from tasks_completed
        ner_annotations1, re_annotations1 = read_annotation_json(tasks_completed_dir, specific_id, chunk_id)
        model_NER_annotations += ner_annotations1
        model_RE_annotations += re_annotations1


        # Read ground truth annotations from manual_annotation_transformed
        gt_ner_annotations, gt_re_annotations = read_annotation_json(manual_transformed_dir, specific_id, chunk_id)
        gt_NER_annotations += gt_ner_annotations
        gt_RE_annotations += gt_re_annotations

        # Precise match NER
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     precise_NER_annotation_match)
        print(f"All Entities - NER Precision: {precision}, NER Recall: {recall}, NER F1 Score: {f1_score}")

        # Approximate match NER
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations, approximate_NER_annotation_match)
        print(f"All Entities - Approximate NER Precision: {precision}, Approximate NER Recall: {recall}, Approximate NER F1 Score: {f1_score}")

        # Precise match RE
        precision, recall, f1_score = RE_evaluation(model_RE_annotations, gt_RE_annotations,
                                                     precise_RE_annotation_match)
        print(f"All Entities - RE Precision: {precision}, RE Recall: {recall}, RE F1 Score: {f1_score}")

        # Approximate match RE
        precision, recall, f1_score = RE_evaluation(model_RE_annotations, gt_RE_annotations,
                                                     approximate_RE_annotation_match)
        print(
            f"All Entities - Approximate RE Precision: {precision}, Approximate RE Recall: {recall}, Approximate RE F1 Score: {f1_score}")

   # Single labels

        # Precise match NER Scientific Plant Name
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     precise_NER_annotation_match, 'Scientific Plant Name')
        print(f"Scientific Plant Name - NER Precision: {precision}, NER Recall: {recall}, NER F1 Score: {f1_score}")

        # Approximate match NER Scientific Plant Name
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     approximate_NER_annotation_match, 'Scientific Plant Name')
        print(
            f"Scientific Plant Name - Approximate NER Precision: {precision}, Approximate NER Recall: {recall}, Approximate NER F1 Score: {f1_score}")

        # Precise match NER Scientific Fungus Name
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     precise_NER_annotation_match, 'Scientific Fungus Name')
        print(f"Scientific Fungus Name - NER Precision: {precision}, NER Recall: {recall}, NER F1 Score: {f1_score}")

        # Approximate match NER Scientific Fungus Name
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     approximate_NER_annotation_match, 'Scientific Fungus Name')
        print(
            f"Scientific Fungus Name - Approximate NER Precision: {precision}, Approximate NER Recall: {recall}, Approximate NER F1 Score: {f1_score}")

        # Precise match NER Medical condition
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     precise_NER_annotation_match, 'Medical Condition')
        print(f"Medical Condition - NER Precision: {precision}, NER Recall: {recall}, NER F1 Score: {f1_score}")

        # Approximate match NER Medical Condition
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     approximate_NER_annotation_match, 'Medical Condition')
        print(
            f"Medical Condition - Approximate NER Precision: {precision}, Approximate NER Recall: {recall}, Approximate NER F1 Score: {f1_score}")

        # Precise match NER Medicinal Effect
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     precise_NER_annotation_match, 'Medicinal Effect')
        print(f"Medicinal Effect - NER Precision: {precision}, NER Recall: {recall}, NER F1 Score: {f1_score}")

        # Approximate match NER Medical Effect
        precision, recall, f1_score = NER_evaluation(model_NER_annotations, gt_NER_annotations,
                                                     approximate_NER_annotation_match, 'Medicinal Effect')
        print(
            f"Medicinal Effect - Approximate NER Precision: {precision}, Approximate NER Recall: {recall}, Approximate NER F1 Score: {f1_score}")



if __name__ == '__main__':
    main()
