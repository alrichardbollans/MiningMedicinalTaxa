# To evaluate NER. Check taxonerd NER outputs.
# recommended instead of taxonerd (See (https://github.com/nleguillarme/taxonerd/issues/15)), but we do remove non sci names now so can check this.

import os
import pickle

import pandas as pd
from pkg_resources import resource_filename

from LLM_models.evaluating.run_evaluation import assess_model_on_chunk_list
from LLM_models.loading_files import read_file_and_chunk
from LLM_models.running_models import get_input_size_limit
from LLM_models.structured_output_schema import TaxaData, Taxon, deduplicate_and_standardise_output_taxa_lists
from useful_string_methods import clean_strings

_inputs_path = resource_filename(__name__, 'inputs')

_temp_outputs_path = resource_filename(__name__, 'temp_outputs')

_output_path = resource_filename(__name__, 'outputs')


class NERModel:
    def __init__(self, model_name):
        self.model_name = model_name


def query_taxonerd(model, text_file: str, context_window: int, pkl_dump: str = None, single_chunk: bool = True) -> TaxaData:
    # text_chunks = read_file_and_chunk(text_file, context_window)
    # names = []
    # for text in text_chunks:
    #     taxonerd_on_chunk = taxonerd.find_in_text(text)
    #     names += taxonerd_on_chunk['text'].tolist()
    # names = set(names)
    names = taxonerd.find_in_file(text_file)['text'].unique().tolist()
    taxa_list = []
    for n in names:
        taxa_list.append(Taxon(scientific_name=n))

    deduplicated_extractions = deduplicate_and_standardise_output_taxa_lists(taxa_list)

    if pkl_dump:
        with open(pkl_dump, "wb") as file_:
            pickle.dump(deduplicated_extractions, file_)

    return deduplicated_extractions


def full_evaluation(model, rerun:bool=True):
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    assess_model_on_chunk_list(test['id'].unique().tolist(), model, None, os.path.join('outputs', 'full_eval'), rerun=rerun,
                               model_query_function=query_taxonerd, autoremove_non_sci_names=False)

    assess_model_on_chunk_list(test['id'].unique().tolist(), model, None, os.path.join('outputs', 'full_eval'), rerun=False,
                               model_query_function=query_taxonerd, autoremove_non_sci_names=True)
    # TODO: Check RE metrics =0


if __name__ == '__main__':
    from taxonerd import TaxoNERD

    taxonerd = TaxoNERD()

    taxonerd_ner_model = taxonerd.load(model="en_ner_eco_biobert")
    taxon_test_model = NERModel("en_ner_eco_biobert")
    full_evaluation(taxon_test_model, rerun=False)
