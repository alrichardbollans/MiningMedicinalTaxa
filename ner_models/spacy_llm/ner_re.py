import os
import json
import logging
from spacy_llm.util import assemble
import spacy_llm
from dotenv import load_dotenv
import shutil
import time

def load_environment_variables():
    """ Load environment variables from a .env file. """
    load_dotenv()

def configure_logging():
    """ Configure logging for the application. """
    handler = logging.StreamHandler()
    spacy_llm.logger.setLevel(logging.DEBUG)
    spacy_llm.logger.addHandler(handler)

def load_spacy_model(config_file):
    """ Load and return a spaCy model based on the provided configuration. """
    return assemble(config_file)

def format_document_spans(doc, entity_id_map):
    """ Format document spans for Label Studio. """
    formatted_spans = []
    for ent in doc.ents:
        entity_id = entity_id_map.get(ent, None)
        if entity_id:
            span_data = {
                'id': entity_id,
                'from_name': 'label',
                'to_name': 'text',
                'type': 'labels',
                'value': {
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'labels': [ent.label_]
                }
            }
            formatted_spans.append(span_data)
    return formatted_spans

def format_document_relations(doc):
    """ Extract and format relations from the document. """
    formatted_relations = []
    entity_id_map = {ent: f"entity_{i}" for i, ent in enumerate(doc.ents)}
    valid_relations = ["treats_medical_condition", "has_medicinal_effect"]

    for rel in doc._.rel:
        if rel.relation in valid_relations:
            relation_data = {
                "from_id": entity_id_map[doc.ents[rel.dep]],
                "to_id": entity_id_map[doc.ents[rel.dest]],
                "type": "relation",
                "labels": [rel.relation]
            }
            formatted_relations.append(relation_data)

    return formatted_relations


def process_text_file(input_folder, output_folder, file, model):
    file_path = os.path.join(input_folder, file)
    with open(file_path, 'r', encoding='utf-8') as file_obj:
        text = file_obj.read()
    doc = model(text)
    print(f"DOC ENTS: {doc.ents}")
    print([(ent.text, ent.label_) for ent in doc.ents])
    llm_io_data = doc.user_data["llm_io"]
    for component_name, io_data in llm_io_data.items():
        print(f"Component: {component_name}")
        #print(f"Prompt: {io_data['prompt']}")
        print(f"Response: {io_data['response']}")
    entity_id_map = {ent: f"entity_{i}" for i, ent in enumerate(doc.ents)}
    entities = format_document_spans(doc, entity_id_map)
    relations = format_document_relations(doc)
    task = {'data': {'text': text}, 'predictions': [{'result': entities + relations}]}

    # Create the chunk name by removing '.txt' from the filename, regardless of its position
    chunk_name = file.replace('.txt', '')

    output_file = f"task_for_labelstudio_{chunk_name}.json"
    save_tasks_to_json([task], output_folder, output_file)

    # Move the processed file to the 'selected_preprocessed_completed' directory
    completed_folder = os.path.join(input_folder, '..', 'selected_preprocessed_completed')
    if not os.path.exists(completed_folder):
        os.makedirs(completed_folder)
    completed_file_path = os.path.join(completed_folder, file)
    shutil.move(file_path, completed_file_path)
    time.sleep(60) # After moving a file sleep for 5 seconds

def save_tasks_to_json(tasks, output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    print(f'Saved tasks to "{output_path}"')

def ner_re_gpt(input_folder='selected_preprocessed', output_folder='task_completed'):
    load_environment_variables()
    configure_logging()
    nlp_model = load_spacy_model('zeroshot.cfg')

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in files:
        process_text_file(input_folder, output_folder, file, nlp_model)


if __name__ == "__main__":
    ner_re_gpt()
