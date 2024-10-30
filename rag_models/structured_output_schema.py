import json
import os
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, Field

from rag_models.rag_prompting import medicinal_effect_def, medical_condition_def
from useful_string_methods import clean_strings, TAXON_ENTITY_CLASSES, get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations, \
    check_human_annotations, ENTITY_CLASSES

repo_path = os.environ.get('KEWSCRATCHPATH')
base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')
base_chunk_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'all_chunks')
annotation_folder = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'annotations',
                                 'manually_annotated_chunks')
_annotation_file = os.path.join(annotation_folder, 'task_for_labelstudio_completed_updated.json')

annotation_info = pd.read_excel(os.path.join(annotation_folder, 'annotated_chunks_list.xlsx'))


def get_corpus_id_from_chunk_name(chunk_name: str) -> str:
    # Get corpus id from a string like: 'task_for_labelstudio_{corpus_id}_chunk_{chunk_id}.json'
    corpus_id = chunk_name.split('task_for_labelstudio_')[1]
    corpus_id = corpus_id.split('_chunk_')[0]
    return corpus_id


annotation_info['corpus_id'] = annotation_info['name'].apply(get_corpus_id_from_chunk_name)

assert annotation_info['reference_only'].unique().tolist() == ['no', 'yes']
valid_chunk_annotation_info = annotation_info[annotation_info['reference_only'] != 'yes']


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
    taxa: Optional[List[Taxon]]


def deduplicate_and_standardise_output_taxa_lists(taxa: List[Taxon]) -> TaxaData:
    """ Clean strings, as in read_annotation_json and then deduplicate results"""
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
                for condition in taxon.medical_conditions or []:
                    if condition == condition and condition.lower() != 'null':
                        new_taxon.medical_conditions.append(condition)
                for effect in taxon.medicinal_effects or []:
                    if effect == effect and effect.lower() != 'null':
                        new_taxon.medicinal_effects.append(effect)

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
    """
    Convert human annotations to taxa data schema.

    :param human_ner_annotations: List of human named entity annotations.
    :param human_re_annotations: List of human relation extraction annotations.
    :return: Instance of TaxaData.
    """
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
        else:
            if entry['value']['label'] not in ENTITY_CLASSES:
                raise ValueError(f"{entry['value']['label']} not in {ENTITY_CLASSES}")
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


def get_chunk_filepath_from_chunk_id(chunk_id: int):
    name = annotation_info[annotation_info['id'] == chunk_id]['name'].iloc[0]
    name = name.removeprefix('task_for_labelstudio_')
    idx = name.index('_chunk')
    name = name[:idx] + '.txt' + name[idx:]
    name = name.replace('.json', '.txt')
    return os.path.join(base_chunk_path, name)


def _get_result_from_ann_in_dict(ann: dict):
    anns = ann['annotations'][0]['result']
    if len(ann['annotations']) > 1:
        print(ann)
        raise ValueError
    chunk_id_for_annotations = ann['id']
    assert ann['id'] == ann['inner_id']

    return anns, chunk_id_for_annotations


def get_all_human_annotations_for_corpus_id(corpus_id: str, check: bool = True):
    """
    For a given corpus_id, get related cleaned and deduplicated human annotations.
    :param corpus_id:
    :param check:
    :return:
    """
    collected_taxa_data = []

    relevant_annotation_info = annotation_info[annotation_info['corpus_id'] == corpus_id]
    relevant_ids = relevant_annotation_info['id'].unique().tolist()

    with open(_annotation_file) as f:
        d = json.load(f)
    for ann in d:
        anns, chunk_id = _get_result_from_ann_in_dict(ann)
        if chunk_id in relevant_ids:
            human_ner_annotations1, human_re_annotations1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns,
                                                                                                                                          check=check)
            taxa_data = convert_human_annotations_to_taxa_data_schema(human_ner_annotations1, human_re_annotations1)
            collected_taxa_data.extend(taxa_data.taxa)
    return deduplicate_and_standardise_output_taxa_lists(collected_taxa_data)


def get_all_human_annotations_for_chunk_id(chunk_id: int, check: bool = True):
    """
    For a given chunk id, get related cleaned and deduplicated human annotations.
    :param chunk_id:
    :param check:
    :return:
    """
    collected_taxa_data = []

    with open(_annotation_file) as f:
        d = json.load(f)
    for ann in d:
        anns, ann_chunk_id = _get_result_from_ann_in_dict(ann)

        if ann_chunk_id == chunk_id:
            human_ner_annotations1, human_re_annotations1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns,
                                                                                                                                          check=check)
            taxa_data = convert_human_annotations_to_taxa_data_schema(human_ner_annotations1, human_re_annotations1)
            collected_taxa_data.extend(taxa_data.taxa)
    if len(collected_taxa_data) == 0:
        print(f'Warning: No human annotations for id: {chunk_id}')
    return deduplicate_and_standardise_output_taxa_lists(collected_taxa_data)


def _check_ann_id_matches_with_info(ann):
    ## Check chunk id matches with info
    chunk_id_for_annotations = ann['id']
    assert ann['id'] == ann['inner_id']
    data = ann['data']['text']

    import os

    with open(os.path.join(get_chunk_filepath_from_chunk_id(chunk_id_for_annotations)), "r", encoding="utf8") as f:
        text = f.read()
    assert data == text


def check_all_human_annotations():
    """
    Validates human annotations by checking each annotation chunk against predetermined criteria and identifying errors.

    Reads annotation data from the specified file, processes each annotation chunk, and verifies its validity.
    If errors or issues are found within the annotations, they are recorded and saved into a CSV file named 'humman_annotation_issues.csv'.

    :raises FileNotFoundError: If the specified annotation file does not exist.
    :raises JSONDecodeError: If the annotation file is not a valid JSON.

    :return: None
    """
    bad_ids = []
    bad_messages = []
    with open(_annotation_file) as f:
        d = json.load(f)
    for ann in d:
        anns, ann_chunk_id = _get_result_from_ann_in_dict(ann)
        _check_ann_id_matches_with_info(ann)
        if ann_chunk_id not in valid_chunk_annotation_info['id'].values:
            try:
                assert len(anns) == 0
            except AssertionError:
                print(f'Reference only chunk {ann_chunk_id} has annotations.')
        try:
            human_ner_annotations1, human_re_annotations1 = get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns,
                                                                                                                                          check=False)
        except Exception as e:
            print(f'Annotation Chunk ID: {ann_chunk_id}. Error message: {e}')
        else:
            try:
                check_human_annotations(human_ner_annotations1, human_re_annotations1)
            except Exception as e:
                print(f'Annotation Chunk ID: {ann_chunk_id}. Error message: {e}')
                bad_ids.append(ann_chunk_id)
                bad_messages.append(e)
    issues = annotation_info[annotation_info['id'].isin(bad_ids)]
    issues['message'] = bad_messages
    if len(issues) > 0:
        issues.to_csv('humman_annotation_issues.csv', index=False)


def summarise_annotations(chunk_ids: list, out_path: str):
    number_of_chunks = len(chunk_ids)
    number_of_taxa = 0
    number_of_lone_taxa = 0
    number_of_medical_conditions = 0
    number_of_medicinal_effects = 0

    for chunk_id in chunk_ids:
        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        taxa = human_annotations.taxa
        number_of_taxa += len(taxa)

        for t in taxa:
            if t.medicinal_effects is None and t.medical_conditions is None:
                number_of_lone_taxa += 1
            for m in t.medical_conditions or []:
                number_of_medical_conditions += 1
            for m in t.medicinal_effects or []:
                number_of_medicinal_effects += 1
    out_df = pd.DataFrame([[number_of_chunks, number_of_taxa, number_of_lone_taxa, number_of_medical_conditions, number_of_medicinal_effects]],
                          columns=['Number of Chunks', 'Taxa', 'Lone Taxa', 'Medical Conditions', 'Medicinal Effects'])
    out_df.to_csv(out_path)


if __name__ == '__main__':
    check_all_human_annotations()
