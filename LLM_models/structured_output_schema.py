from typing import Optional, List
from pydantic import BaseModel, Field, Extra

from LLM_models.prompting import medicinal_effect_def, medical_condition_def
from useful_string_methods import clean_strings, TAXON_ENTITY_CLASSES, get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations, \
    check_human_annotations, ENTITY_CLASSES


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


class TaxaData(BaseModel, extra=Extra.allow):
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
