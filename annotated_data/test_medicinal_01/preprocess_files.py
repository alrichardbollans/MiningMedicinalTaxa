import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ner_models', 'spacy_llm'))

from ner_models.spacy_llm import (download_json_from_coreid,
                                  process_json_files, process_files, get_chunks)


file_path = 'coreid_to_download.json'

with open(file_path, 'r') as file:
 coreids = json.load(file)
 coreid_list = coreids['core_ids']

known_issues = ["error_id_1", "error_id_2"]  # example for ID know to have issue (maybe remove this)
directory = '10_medicinal_hits_json'  # Store Json files

if not os.path.exists(directory):
    os.makedirs(directory)

download_json_from_coreid(coreid_list, known_issues, directory)

# Specify your directories here
input_directory = '10_medicinal_hits_json'
output_directory = '10_medicinal_hits'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_json_files(input_directory, output_directory)



# Set your input and output directories here
preprocess_input_folder = '10_medicinal_hits'
preprocess_output_folder = 'preprocessed'
if not os.path.exists(preprocess_output_folder):
    os.makedirs(preprocess_output_folder)
# Call the function directly with the specified folders
process_files(preprocess_input_folder, preprocess_output_folder)

# Extract keywords
# Example: Modify these variables to match your requirements or read them from command-line arguments
csv_path = 'medicinals_top_10000.csv'  # Path to your CSV file
preprocessed_dir = 'preprocessed'  # Directory containing preprocessed files

selected_preprocessed_dir = 'selected_preprocessed'  # Directory to save selected files
if not os.path.exists(selected_preprocessed_dir):
    os.makedirs(selected_preprocessed_dir)
subset_size = 10  # Number of records to process

get_chunks(csv_path, preprocessed_dir, selected_preprocessed_dir, subset_size)
