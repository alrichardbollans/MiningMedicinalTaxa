import json
from pathlib import Path
import os

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def check_and_correct_mismatches(data):
    """Check and correct mismatches between outer id, task, and first annotation id."""
    mismatches = []
    current_annotation_id = 1  # Starting annotation ID expected

    for idx, item in enumerate(data):
        expected_id = idx + 1
        annotation_id = current_annotation_id

        # Record mismatch if found
        if item['id'] != expected_id or item['annotations'][0]['id'] != annotation_id:
            mismatches.append({
                'outer_id': item['id'],
                'first_annotation_id': item['annotations'][0]['id'],
                'task_id': item['annotations'][0].get('task', None),
                'import_id': item['annotations'][0].get('import_id', None),
                'item_index': idx
            })

        # Correct outer id, task id, and first annotation id
        item['id'] = expected_id
        item['annotations'][0]['id'] = annotation_id
        item['annotations'][0]['task'] = annotation_id
        item['annotations'][0]['import_id'] = annotation_id

        current_annotation_id += 1  # Increment annotation ID

    return data, mismatches


def main():
    # Define file paths
    base_path = Path(
        os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations',
                     'manually_annotated_chunks'))

    input_file = base_path / 'task_for_labelstudio_completed.json'
    output_file = base_path / 'task_for_labelstudio_completed_updated.json'
    mismatches_file = base_path / 'mismatches_report.json'

    # Load the JSON data
    data = load_json_file(input_file)

    # Check and correct mismatches
    corrected_data, mismatches = check_and_correct_mismatches(data)

    # Save the corrected JSON data
    save_json_file(corrected_data, output_file)
    print(f"Corrected JSON file saved to {output_file}")

    # Save the mismatch report
    save_json_file(mismatches, mismatches_file)
    print(f"Mismatch report saved to {mismatches_file}")


if __name__ == '__main__':
    main()
