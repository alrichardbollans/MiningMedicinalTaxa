import os
import json
import logging
from spacy_llm.util import assemble
from itertools import groupby
import spacy_llm


# Create a logging handler and configure logging (useful to visualise the spacy prompt)
handler = logging.StreamHandler()
spacy_llm.logger.setLevel(logging.DEBUG)
spacy_llm.logger.addHandler(handler)

# Load the spaCy LLM model
nlp = assemble("config.cfg")

# Function to format document spans for Label Studio
def doc_to_spans(doc):
    tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
    results = []
    for entity, group in groupby(tokens, key=lambda t: t[-1]):
        if not entity:
            continue
        group = list(group)
        _, start, _ = group[0]
        word, last, _ = group[-1]
        end = last + len(word)
        results.append({
            'from_name': 'label',
            'to_name': 'text',
            'type': 'labels',
            'value': {
                'start': start,
                'end': end,
                'text': word,
                'labels': [entity]
            }
        })
    return results

# Path to the folder containing preprocessed files
preprocessed_folder = 'preprocessed'
tasks = []

# Process each file in the preprocessed folder
for file in os.listdir(preprocessed_folder):
    if file.endswith('.txt'):
        file_path = os.path.join(preprocessed_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = nlp(text)
        formatted_entities = doc_to_spans(doc)

        # Prepare a task in import JSON format for Label Studio
        task = {
            'data': {'text': text},
            'predictions': [{
                'result': formatted_entities
            }]
        }
        tasks.append(task)

# Save the tasks to a JSON file for Label Studio
output_file = 'tasks_for_label_studio.json'
with open(output_file, 'w', encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)  # The tasks are wrapped in a list

print(f'Saved {len(tasks)} tasks to "{output_file}"')
