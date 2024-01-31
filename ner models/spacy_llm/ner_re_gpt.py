import os
import json
import logging
import spacy_llm
from spacy_llm.util import assemble
from dotenv import load_dotenv


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


def process_text_files(folder_path, model):
    """ Process text files in a given folder and extract entities and relations. """
    tasks = []
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as file_obj:
                text = file_obj.read()
                doc = model(text)
                entity_id_map = {ent: f"entity_{i}" for i, ent in enumerate(doc.ents)}
                entities = format_document_spans(doc, entity_id_map)
                relations = format_document_relations(doc)
                task = {'data': {'text': text}, 'predictions': [{'result': entities + relations}]}
                tasks.append(task)
    return tasks


def save_tasks_to_json(tasks, output_file):
    """ Save the tasks to a JSON file. """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    print(f'Saved {len(tasks)} tasks to "{output_file}"')


def main():
    load_environment_variables()
    configure_logging()
    nlp_model = load_spacy_model('zeroshot.cfg')
    preprocessed_folder = 'preprocessed'
    tasks = process_text_files(preprocessed_folder, nlp_model)
    save_tasks_to_json(tasks, 'tasks_for_label_studio.json')


if __name__ == "__main__":
    main()
