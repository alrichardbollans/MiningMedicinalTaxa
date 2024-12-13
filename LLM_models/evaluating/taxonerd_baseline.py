# To evaluate NER. Check taxonerd NER outputs.

import os
import pickle

import pandas as pd
from pkg_resources import resource_filename

from LLM_models.evaluating.run_zeroshot_evaluation import assess_model_on_chunk_list
from LLM_models.structured_output_schema import TaxaData, Taxon
from useful_string_methods import clean_strings

_inputs_path = resource_filename(__name__, 'inputs')

_temp_outputs_path = resource_filename(__name__, 'temp_outputs')

_output_path = resource_filename(__name__, 'outputs')


class NERModel:
    def __init__(self, model_name):
        self.model_name = model_name


def make_taxaData_from_output(taxonerd_output) -> TaxaData:
    unique_scientific_names = []
    for taxon in taxonerd_output['text'].values:
        clean_name = clean_strings(taxon)
        if clean_name not in unique_scientific_names:
            unique_scientific_names.append(clean_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, medical_conditions=[], medicinal_effects=[])
        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def query_taxonerd(model, text_file: str, context_window: int, pkl_dump: str = None, single_chunk: bool = True) -> TaxaData:
    t_output = taxonerd.find_in_file(text_file)
    deduplicated_extractions = make_taxaData_from_output(t_output)

    if pkl_dump:
        with open(pkl_dump, "wb") as file_:
            pickle.dump(deduplicated_extractions, file_)

    return deduplicated_extractions


def full_evaluation():
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    assess_model_on_chunk_list(test['id'].unique().tolist(), taxon_test_model, 99999999999999999, 'outputs',
                               model_query_function=query_taxonerd, autoremove_non_sci_names=False)

    assess_model_on_chunk_list(test['id'].unique().tolist(), taxon_test_model, 99999999999999999, 'outputs',
                               model_query_function=query_taxonerd, autoremove_non_sci_names=True)
    # TODO: Check RE metrics =0



if __name__ == '__main__':
    from taxonerd import TaxoNERD

    taxonerd = TaxoNERD(prefer_gpu=False)

    # TODO: Fix entity linking. See (https://github.com/nleguillarme/taxonerd/issues/15)
    # taxonerd_ner_model = taxonerd.load(model="en_ner_eco_biobert", exclude=[], linker='gbif_backbone')
    # taxon_test_model = NERModel("en_ner_eco_biobert_gbif_linker")
    # full_evaluation()

    taxonerd_ner_model = taxonerd.load(model="en_ner_eco_biobert", exclude=[])
    taxon_test_model = NERModel("en_ner_eco_biobert")
    full_evaluation()
