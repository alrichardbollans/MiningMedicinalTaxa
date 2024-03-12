import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ner_models', 'spacy_llm'))

from ner_models.spacy_llm import ner_re_gpt

input_folder = 'selected_preprocessed'
output_file = 'task_for_label_studio_286439500_1.json'  # Adjust this file path
ner_re_gpt(input_folder, output_file)
