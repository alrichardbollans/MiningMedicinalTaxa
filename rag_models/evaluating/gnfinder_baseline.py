# cite with https://zenodo.org/records/11584025
# recommended instead of taxonerd (See (https://github.com/nleguillarme/taxonerd/issues/15))
# see $HOME/.config/gnfinder.yml for configs (have left defaults).
import os
import pickle
import subprocess
import time
from io import BytesIO

import pandas as pd

from rag_models.structured_output_schema import Taxon, deduplicate_and_standardise_output_taxa_lists


def get_gnfinder_verbatim_names_from_text_file(text_file: str):
    time.sleep(0.1)
    command = "gnfinder {text_file}".format(text_file=text_file)

    output = subprocess.check_output(command, shell=True)
    # Use BytesIO to treat the bytes as a file
    byte_stream = BytesIO(output)

    # Read it into a Pandas DataFrame
    df = pd.read_csv(byte_stream)
    return df['Verbatim'].unique().tolist()


def get_taxa_data_from_gnfinder_on_text_file(text_file: str):
    names = get_gnfinder_verbatim_names_from_text_file(text_file)
    taxa_list = []
    for n in names:
        taxa_list.append(Taxon(scientific_name=n))

    out_taxa = deduplicate_and_standardise_output_taxa_lists(taxa_list)

    return out_taxa


def gnfinder_query_function(model, text_file: str, context_window: int, pkl_dump: str):
    out_taxa_data = get_taxa_data_from_gnfinder_on_text_file(text_file)

    with open(pkl_dump, "wb") as file_:
        pickle.dump(out_taxa_data, file_)
    return out_taxa_data


if __name__ == '__main__':
    # get_taxa_data_from_gnfinder_on_text(text="Pomatomus saltator and Parus major")
    repo_path = os.environ.get('KEWSCRATCHPATH')
    base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')

    example_model_outputs = gnfinder_query_function(None, os.path.join(base_text_path, '4187756.txt'), None, 'gn_pkl.pkl')
