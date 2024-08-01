import sys
import os
from dotenv import load_dotenv

load_dotenv()
CORE_API_KEY = os.getenv('CORE_API_KEY')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre_labelling', 'spacy.llm'))

from pre_labelling.spacy_llm.download_and_clean_txt import process_and_save_text
from pre_labelling.spacy_llm.process_and_chunk_txt import process_and_chunk_text
from pre_labelling.spacy_llm.process_and_chunk_txt import select_chunks

coreid_list = ["80818116", "286439500", "360558516", "228197190", "268329601", "197989253", "4187756", "481470282", "161880242", "35321774"]

known_issues = "error_id_1, error_id_2"  # Example for IDs known to have issues
output_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'text_files')  # Directory to store the text files

# Process each core ID
for coreid in coreid_list:
    process_and_save_text(coreid, known_issues, output_directory)

# Process each txt file and split in chunks of 4000 tokens

input_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'text_files')
output_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'all_chunks')

process_and_chunk_text(input_directory, output_directory)

# Select chunk that match keywords in medicinal_top_10000

csv_path = 'C:\\Users\\fci10kg\\OneDrive - The Royal Botanic Gardens, Kew\\Documents\\mygithub\\MedicinalPlantMining\\literature_downloads\\core\\downloads\\medicinals_top_10000\\medicinals_top_10000.csv'
input_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'all_chunks')
output_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'selected_chunks')
log_path = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'chunks')
subset_size = 10  # Number of records to process

select_chunks(csv_path, input_directory, output_directory, subset_size, log_path)