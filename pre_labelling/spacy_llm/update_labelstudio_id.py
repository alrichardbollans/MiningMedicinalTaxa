import json
import os
from pathlib import Path


def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json_file(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def update_task_ids(data):
    """Replace the task ID with the inner annotation ID in the given data,
       and fix the annotation IDs starting from a specific number."""

    next_annotation_id = 322  # Start fixing annotation IDs from 322

    for item in data:
        if 'annotations' in item and item['annotations']:
            # Update inner annotation ID if it should follow a corrected sequence
            for annotation in item['annotations']:
                annotation['id'] = next_annotation_id
                annotation['task'] = next_annotation_id
                next_annotation_id += 1

            # Replace task ID with the first annotation ID
            item['id'] = item['annotations'][0]['id']

    return data


def main():
    # Define the base path relative to this script's location
    base_path = Path(
        os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations',
                     'manually_annotated_chunks'))

    # Specify input and output files
    input_file = base_path / 'task_for_label_studio_completed.json'
    output_file = base_path / 'task_for_labelstudio_completed_updated.json'

    # Load the JSON data
    data = load_json_file(input_file)

    # Update task IDs and annotation IDs
    updated_data = update_task_ids(data)

    # Save the updated JSON data
    save_json_file(updated_data, output_file)
    print(f"Updated JSON file saved to {output_file}")


if __name__ == '__main__':
    main()
