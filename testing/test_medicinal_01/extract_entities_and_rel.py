import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ner_models', 'spacy_llm'))

from ner_models.spacy_llm import ner_re_gpt

input_folder = 'selected_preprocessed'
output_folder = 'tasks_completed'
ner_re_gpt(input_folder, output_folder)