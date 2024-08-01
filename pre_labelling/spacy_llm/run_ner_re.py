import os

from pre_labelling.spacy_llm import ner_re_gpt

input_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'selected_chunks')
output_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations', 'pre_labelled_chunks')
log_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'annotated_data', 'top_10_medicinal_hits', 'annotations', 'pre_labelled_chunks.log')
config_file = 'zeroshot.cfg'
ner_re_gpt(input_directory, output_directory, log_file_path, config_file)

