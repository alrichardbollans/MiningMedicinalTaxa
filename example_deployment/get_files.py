import json
import pickle
import shutil
import time
from collections import defaultdict

import pandas as pd
import requests
from requests import HTTPError

from LLM_models.checking_and_summarising_annotations import repo_path
from pre_labelling.spacy_llm import process_and_chunk_text

api_endpoint = "https://api.core.ac.uk/v3/"

# Some pkls to store info about previous searches
request_pkl_file = 'doi_request_dict.pkl'
try:
    with open(request_pkl_file, 'rb') as _pfile:
        _request_dict_info = pickle.load(_pfile)
except FileNotFoundError:
    _request_dict_info = {}

with open('secrets.txt') as keyfile:
    _core_apikey = keyfile.read()

def extract_info(hit: dict) -> dict:
    """
    Extract specific information from a dictionary and organize it into a new dictionary.

    This function iterates over a set of predefined keys to extract their values from the
    input dictionary. If the key does not exist in the input dictionary, it skips without
    throwing an error. Additionally, it attempts to extract a nested value for the language
    key if present.

    :param hit: The input dictionary containing data to be extracted.
    :type hit: dict
    :return: A dictionary containing extracted key-value pairs. If a key is not found in
             the input dictionary, it is omitted. For the 'language' key, the nested
             'code' value is extracted if available.
    :rtype: dict
    """
    out_dict = defaultdict(None)
    for key in ['id', 'downloadUrl', 'doi', 'fullText', 'sourceFulltextUrls']:
        try:
            out_dict[key] = hit[key]
        except KeyError:
            pass
    try:
        out_dict['language'] = hit["language"]['code']
    except (KeyError, TypeError):
        pass
    return out_dict

def get_results_for_corpusid(corpusid: str) -> dict:
    """
    Fetches results associated with a given corpusid by querying an external API.
    The function handles rate-limiting scenarios and retries requests if necessary. Results are cached locally
    for future lookups.

    :param corpusid: The Digital Object Identifier for which results are to be fetched.
    :type corpusid: str
    :return: A list of dictionaries containing extracted information for the given corpusid.
    :rtype: list[dict]
    """

    if corpusid in _request_dict_info:
        return _request_dict_info[corpusid]
    # else:
    #     raise HTTPError(f'Error getting results for {doi}')

    time.sleep(2)
    headers = {"Authorization": "Bearer " + _core_apikey}

    # NOte this params method is commented out as it seems to be broken, although its following the example
    # here: https://api.core.ac.uk/docs/v3#tag/Works/operation/optionsCustomSearchWorks
    # params = {
    #     'q': f'doi:"{doi}"',
    # }
    # response = requests.get(f"{api_endpoint}works", headers=headers, params=params)
    response = requests.get(f"{api_endpoint}outputs/{corpusid}", headers=headers)

    if response.status_code == 429:
        # retry
        time.sleep(10)  # Rate limiting
        response = requests.get(f"{api_endpoint}outputs/{corpusid}", headers=headers)
    if response.status_code == 200:

        result = response.json()

        out_dict = extract_info(result)

        _request_dict_info[corpusid] = out_dict
        with open(request_pkl_file, 'wb') as pfile:
            pickle.dump(_request_dict_info, pfile)
        # print(request_dict_info)
        return out_dict

    elif response.status_code == 429:
        raise ValueError('Rate limit exceeded')
    elif response.status_code == 500:
        raise HTTPError(f'Error getting results for {corpusid}. response code: {response.status_code}')
    elif response.status_code == 410:
        raise PermissionError(f'Error getting results for {corpusid}. response code: {response.status_code}, text: {response.text}')
    else:
        raise ValueError(f'Something else is wrong. response code: {response.status_code}')

def prepare_data():
    top_15_hits = pd.read_csv(
        f'{repo_path}/MedicinalPlantMining/literature_downloads/core/downloads/medicinals_top_10000/medicinals_top_10000.csv', index_col=0).head(15)

    top_10_hits = pd.read_csv(
        f'{repo_path}/MedicinalPlantMining/literature_downloads/core/downloads/medicinals_top_10/medicinals_top_10.csv', index_col=0)

    top10_to_15_hits = top_15_hits[~top_15_hits['corpusid'].isin(top_10_hits['corpusid'].unique().tolist())]

    assert len(top10_to_15_hits) == 5

    for corpusid in top10_to_15_hits['corpusid']:
        print(corpusid)
        try:
            c_text = get_results_for_corpusid(str(corpusid))['fullText']
            with open(f'text_files/{corpusid}.txt', 'w') as f:
                f.write(c_text)
        except PermissionError as e:
            print(e)
    import nltk
    nltk.download('punkt_tab')
    process_and_chunk_text('text_files', 'chunks')




def main():
    prepare_data()


if __name__ == '__main__':
    main()
