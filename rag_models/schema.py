from typing import Optional, List

from langchain_core.pydantic_v1 import BaseModel, Field


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
        default=None, description="Specific health issues, diseases, or physical states that the taxon is used to treat."
    )
    medicinal_effects: Optional[List[str]] = Field(
        default=None,
        description="Therapeutic or negative effects induced by consuming the taxon, such as 'antitumor', 'anti-inflammatory', or 'digestive stimulant'."
    )


class TaxaData(BaseModel):
    """Extracted data about taxa."""

    # Creates a model so that we can extract multiple entities.
    taxa: List[Taxon]


def dedpulicate_taxa_lists(taxa: List[Taxon]) -> List[Taxon]:
    # TODO: test this
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None and taxon.scientific_name not in unique_scientific_names:
            unique_scientific_names.append(taxon.scientific_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, medical_conditions=[], medicinal_effects=[])
        for taxon in taxa:
            if taxon.scientific_name == name:
                if taxon.medical_conditions is not None:
                    new_taxon.medical_conditions.extend(taxon.medical_conditions)
                if taxon.medicinal_effects is not None:
                    new_taxon.medicinal_effects.extend(taxon.medicinal_effects)

        if len(new_taxon.medical_conditions) == 0:
            new_taxon.medical_conditions = None
        else:
            new_taxon.medical_conditions = list(set(new_taxon.medical_conditions))
        if len(new_taxon.medicinal_effects) == 0:
            new_taxon.medicinal_effects = None
        else:
            new_taxon.medicinal_effects = list(set(new_taxon.medicinal_effects))

        new_taxa_list.append(new_taxon)
    return new_taxa_list


def convert_human_annotations_to_taxa_data_schema(human_ner_annotations, human_re_annotations) -> TaxaData:
    # TODO: test this
    collected_output = {}
    for entry in human_re_annotations:
        # make entry
        if len(entry['from_entity']['value']['labels']) > 1 or len(entry['to_entity']['value']['labels']) > 1:
            raise ValueError
        from_label = entry['from_entity']['value']['labels'][0]
        if from_label not in TAXON_ENTITY_CLASSES:
            raise ValueError
        taxon = entry['from_entity']['value']['text']

        if taxon not in collected_output.keys():
            collected_output[taxon] = {'medical_conditions': [], 'medicinal_effects': []}

        to_label = entry['to_entity']['value']['labels'][0]
        if to_label not in MEDICINAL_CLASSES:
            raise ValueError
        medicinal_property = entry['to_entity']['value']['text']
        if entry['label'] not in RELATIONS:
            raise ValueError
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


if __name__ == '__main__':
    import sys

    sys.path.append('../testing/evaluation_methods/')
    from testing.evaluation_methods import read_annotation_json, TAXON_ENTITY_CLASSES, RELATIONS, MEDICINAL_CLASSES

    human_ner_annotations1, human_re_annotations1 = read_annotation_json('../testing/test_medicinal_01/manual_annotation_transformed', '4187756', '0')

    convert_human_annotations_to_taxa_data_schema(human_ner_annotations1, human_re_annotations1)
