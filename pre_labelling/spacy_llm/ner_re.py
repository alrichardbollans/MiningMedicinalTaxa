import os
import json
import logging
from spacy_llm.util import assemble
import spacy_llm
from dotenv import load_dotenv
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


def extract_entities(input_folder, output_folder, file, model, log_file_path):
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

    # Append the processed file name to the log file
    with open(log_file_path, 'a') as log_file:
        log_file.write(file + '\n')

    time.sleep(60) # After moving a file sleep for 60 seconds

def save_tasks_to_json(tasks, output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    print(f'Saved tasks to "{output_path}"')

def ner_re_gpt(input_folder, output_folder, log_file_path, config_file):
    load_environment_variables()
    configure_logging()
    nlp_model = load_spacy_model(config_file)

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in files:
        extract_entities(input_folder, output_folder, file, nlp_model, log_file_path)


if __name__ == "__main__":
    ner_re_gpt()
