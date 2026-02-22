import json
import pickle
import re

import pandas as pd


from LLM_models.structured_output_schema import TaxaData, get_chunk_filepath_from_chunk_id
from useful_string_methods import abbreviate_sci_name


def load_taxa_data(json_file: str):

    with open(json_file, "r") as file_:
        json_dict = json.load(file_)
        model_annotations = TaxaData.model_validate(json_dict)
    return model_annotations


def get_relations_from_taxa_data(json_file: str, chunk_id: int):
    model_outputs = load_taxa_data(json_file)
    relations = []
    for taxon in model_outputs.taxa:
        for r in taxon.medical_conditions or []:
            relations.append((taxon.scientific_name, r, 'treats_medical_condition', chunk_id))
        for r in taxon.medicinal_effects or []:
            relations.append((taxon.scientific_name, r, 'has_medicinal_effect', chunk_id))

    return relations


def read_chunk_file(chunk_id: int):
    file_path = get_chunk_filepath_from_chunk_id(chunk_id)

    with open(file_path, 'r') as file:
        return file.read()


if __name__ == '__main__':
    load_taxa_data('70_gpt-4o_outputs.json')
    read_chunk_file(70)
    get_relations_from_taxa_data(
        '70_gpt-4o_outputs.json',
        70)
