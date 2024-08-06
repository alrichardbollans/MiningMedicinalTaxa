import json
import os
import sys
from typing import Optional, List

import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field

# TODO: reorganise and check usage of these methods. Should probably be in 'annotated_data' folder
from pre_labelling.evaluating import clean_strings, check_human_annotations, TAXON_ENTITY_CLASSES, \
    get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations

from rag_models.rag_prompting import medicinal_effect_def, medical_condition_def

_repos_path = os.environ.get('KEWSCRATCHPATH')
annotation_folder = os.path.join(_repos_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'annotations',
                                 'manually_annotated_chunks')
annotation_info = pd.read_excel(os.path.join(annotation_folder, 'annotated_chunks_list.xlsx'))
annotation_info = annotation_info[annotation_info['reference_only'] != 'yes']


def get_corpus_id_from_chunk_name(chunk_name: str) -> str:
    # Get corpus id from a string like: 'task_for_labelstudio_{corpus_id}_chunk_{chunk_id}.json'
    corpus_id = chunk_name.split('task_for_labelstudio_')[1]
    corpus_id = corpus_id.split('_chunk_')[0]
    return corpus_id


annotation_info['corpus_id'] = annotation_info['name'].apply(get_corpus_id_from_chunk_name)


class Taxon(BaseModel):
    """Information about a plant or fungus."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Taxon,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    scientific_name: Optional[str] = Field(default=None,
                                           description="The scientific name of the taxon, with scientific authority in the name if it appears in the text.")
    medical_conditions: Optional[List[str]] = Field(
        default=None, description=medical_condition_def
    )
    medicinal_effects: Optional[List[str]] = Field(
        default=None,
        description=medicinal_effect_def
    )


class TaxaData(BaseModel):
    """Extracted data about taxa."""

    # Creates a model so that we can extract multiple entities.
    taxa: List[Taxon]


def deduplicate_and_standardise_output_taxa_lists(taxa: List[Taxon]) -> TaxaData:
    ''' Clean strings, as in read_annotation_json and then deduplicate results'''
    # TODO: also replace standardise all apostrophes?
    # TODO: test this
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None:
            clean_name = clean_strings(taxon.scientific_name)
            if clean_name not in unique_scientific_names:
                unique_scientific_names.append(clean_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, medical_conditions=[], medicinal_effects=[])
        for taxon in taxa:
            if clean_strings(taxon.scientific_name) == name:
                if taxon.medical_conditions is not None:
                    new_taxon.medical_conditions.extend(taxon.medical_conditions)
                if taxon.medicinal_effects is not None:
                    new_taxon.medicinal_effects.extend(taxon.medicinal_effects)

        if len(new_taxon.medical_conditions) == 0:
            new_taxon.medical_conditions = None
        else:
            cleaned_version = [clean_strings(c) for c in new_taxon.medical_conditions]
            new_taxon.medical_conditions = list(set(cleaned_version))
        if len(new_taxon.medicinal_effects) == 0:
            new_taxon.medicinal_effects = None
        else:
            cleaned_version = [clean_strings(c) for c in new_taxon.medicinal_effects]
            new_taxon.medicinal_effects = list(set(cleaned_version))

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def convert_human_annotations_to_taxa_data_schema(human_ner_annotations, human_re_annotations) -> TaxaData:
    # TODO: test this

    collected_output = {}
    for entry in human_re_annotations:
        # make entry
        taxon = entry['from_entity']['value']['text']

        if taxon not in collected_output.keys():
            collected_output[taxon] = {'medical_conditions': [], 'medicinal_effects': []}

        medicinal_property = entry['to_entity']['value']['text']

        if entry['label'] == 'treats_medical_condition':
            if medicinal_property not in collected_output[taxon]['medical_conditions']:
                collected_output[taxon]['medical_conditions'].append(medicinal_property)
        elif entry['label'] == 'has_medicinal_effect':
            if medicinal_property not in collected_output[taxon]['medicinal_effects']:
                collected_output[taxon]['medicinal_effects'].append(medicinal_property)
        else:
            raise ValueError

    for entry in human_ner_annotations:
        if entry['value']['label'] in TAXON_ENTITY_CLASSES:
            taxon = entry['value']['text']
            if taxon not in collected_output.keys():
                # make entry
                collected_output[taxon] = {'medical_conditions': [], 'medicinal_effects': []}

    outlist = []
    for taxon in collected_output.keys():
        # Convert to None in case there are no values as this is the default.
        if collected_output[taxon]['medical_conditions'] == []:
            collected_output[taxon]['medical_conditions'] = None
        if collected_output[taxon]['medicinal_effects'] == []:
            collected_output[taxon]['medicinal_effects'] = None
        outlist.append(Taxon(scientific_name=taxon, medical_conditions=collected_output[taxon]['medical_conditions'],
                             medicinal_effects=collected_output[taxon]['medicinal_effects']))
    out = TaxaData(taxa=outlist)
    return out


def get_all_human_annotations_for_corpus_id(corpus_id: str, check: bool = True):
    collected_taxa_data = []

    annotation_file = os.path.join(annotation_folder, 'task_for_labelstudio_161880242_228197190_268329601_4187556_360558516_80818116.json')

    relevant_annotation_info = annotation_info[annotation_info['corpus_id'] == corpus_id]
    relevant_ids = relevant_annotation_info['id'].unique().tolist()

    with open(annotation_file) as f:
        d = json.load(f)
    for ann in d:
        if ann['id'] in relevant_ids:
            if len(ann['annotations']) > 1:
                raise ValueError
            anns = ann['annotations'][0]['result']
            human_ner_annotations1, human_re_annotations1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns,
                                                                                                                                          check=check)
            taxa_data = convert_human_annotations_to_taxa_data_schema(human_ner_annotations1, human_re_annotations1)
            collected_taxa_data.extend(taxa_data.taxa)
    return deduplicate_and_standardise_output_taxa_lists(collected_taxa_data)


def check_all_human_annotations():
    annotation_file = os.path.join(annotation_folder, 'task_for_labelstudio_161880242_228197190_268329601_4187556_360558516_80818116.json')
    bad_ids = []
    bad_messages = []
    with open(annotation_file) as f:
        d = json.load(f)
    for ann in d:
        if len(ann['annotations']) > 1:
            raise ValueError
        anns = ann['annotations'][0]['result']
        human_ner_annotations1, human_re_annotations1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns,
                                                                                                                                      check=False)
        try:
            check_human_annotations(human_ner_annotations1, human_re_annotations1)
        except Exception as e:
            print(f'Annotation Chunk ID: {ann["id"]}. Error message: {e}')
            bad_ids.append(ann['id'])
            bad_messages.append(e)
    issues = annotation_info[annotation_info['id'].isin(bad_ids)]
    issues['message'] = bad_messages
    issues.to_csv('humman_annotation_issues.csv', index=False)


if __name__ == '__main__':
    check_all_human_annotations()
    # corpus_output = get_all_human_annotations_for_corpus_id('4187756')
