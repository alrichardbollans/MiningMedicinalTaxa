import os
import json


def find_files_with_empty_results(folder_path, output_path):
    empty_result_files = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Check if predictions have empty results
            # Check if data is a list
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "predictions" in item:
                        if any(prediction.get("result") == [] for prediction in item["predictions"]):
                            empty_result_files.append(filename)
                            break

    with open(output_path, 'w') as output_file:
        for file in empty_result_files:
            output_file.write(f"{file}\n")



folder_path = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations', 'pre_labelled_chunks')
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations', "pre_labelled_chunks_with_parsing_issues.txt")

find_files_with_empty_results(folder_path, output_path)

