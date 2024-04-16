import json
from NER_metrics import read_annotation_json, NER_evaluation, precise_NER_annotation_match, approximate_NER_annotation_match

# TODO: loop over task ids to get the total TP FP and FN and calculate overall precision, recall and F1

def read_annotation_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_data(filename):
    """
    Load JSON data from a file.

    Parameters:
    - filename: The name of the file to load data from.

    Returns:
    - The loaded JSON data.
    """
    with open(filename, 'r') as file:
        return json.load(file)

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
    # Load JSON data from file
   # data_filename = '../test_medicinal_01/manual_annotation/task_for_labelstudio_4187756.json'  # Update with your actual filename
    data_filename = '../test_medicinal_01/manual_annotation/task_for_labelstudio_80818116_ifra_FC_09_04_2024.json'  # Update with your actual filename
    data = load_json_data(data_filename)

    # Specify the task ID dynamically (could be user input or another source)
    #task_id_input = 272  # Example task ID
   # task_id_input = 282 # Example task ID

   task_id_to_output = {
        282: 'task_for_labelstudio_80818116_chunk_111.json',
        283: 'task_for_labelstudio_80818116_chunk_112.json',
        # Add more mappings as needed
    }


    # Transform the data for the specified task ID
    transformed_result = transform_annotation_for_task_id(data, task_id_input)

    if transformed_result:
        print(json.dumps(transformed_result, indent=2))

    if transformed_result:
        print(transformed_result)

    if transformed_result:
        #output_filename = '../test_medicinal_01/manual_annotation_transformed/task_for_labelstudio_4187756_chunk_0.json'
        output_filename = '../test_medicinal_01/manual_annotation_transformed/task_for_labelstudio_80818116_chunk_111.json'
        save_transformed_annotation(transformed_result, output_filename)


    #ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '4187756', '0')
    #gt_ner_annotations, gt_re_annotations = read_annotation_json('../test_medicinal_01/manual_annotation_transformed', '4187756', '0')

    ner_annotations, re_annotations = read_annotation_json('../test_medicinal_01/tasks_completed', '80818116', '111')
    gt_ner_annotations, gt_re_annotations = read_annotation_json('../test_medicinal_01/manual_annotation_transformed',
                                                                 '80818116', '111')
    # Precise match
    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match)
    print(f"All Entities - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                                 'Scientific Plant Name')
    print(f"Scientific Plant Name - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, precise_NER_annotation_match,
                                              'Medical Condition')
    print(f"Medical Condition - Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # Approximate match

    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match)
    print(f"All Entities - Approximate Precision: {precision}, Approximate Recall: {recall}, Approximate F1 Score: {f1_score}")

    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                              'Scientific Plant Name')
    print(f"Scientific Plant Name - Approximate Precision: {precision}, Approximate Recall: {recall}, Approximate F1 Score: {f1_score}")

    precision, recall, f1_score = NER_evaluation(ner_annotations, gt_ner_annotations, approximate_NER_annotation_match,
                                               'Medical Condition')
    print(f"Medical Condition - Approximate Precision: {precision}, Approximate Recall: {recall}, Approximate F1 Score: {f1_score}")


if __name__ == '__main__':
    main()
