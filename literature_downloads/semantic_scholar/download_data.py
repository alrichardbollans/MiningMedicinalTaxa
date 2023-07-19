# Shows how to download and inspect data in the sample datasets
# which are much smaller than the full datasets.
import json
import os.path
from typing import List

import pandas as pd

from literature_downloads.filter_terms import search_filtering_terms
from literature_downloads.semantic_scholar import download_path, S2_API_KEY

dataset_download_path = os.path.join(download_path, 'datasets')
abstracts_path = os.path.join(download_path, 'abstracts')
text_path = os.path.join(download_path, 'text')
paper_info_path = os.path.join(download_path, 'paper_info')
if not os.path.exists(dataset_download_path):
    os.mkdir(dataset_download_path)


def get_zip_file_for_part(part: int):
    return os.path.join(dataset_download_path, "s2orc-part" + str(part) + ".jsonl.gz")


def get_unzipped_file_for_part(part: int):
    return os.path.join(dataset_download_path, "s2orc-part" + str(part) + ".jsonl")


def download_fullset():
    import requests
    import urllib
    for part in range(1, 30):
        # Get info about the latest release
        latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
        print(latest_release['README'])
        print(latest_release['release_id'])

        # Print names of datasets in the release
        print("\n".join(d['name'] for d in latest_release['datasets']))

        # Print README for one of the datasets
        print(latest_release['datasets'][2]['README'])

        # Get info about the s2orc dataset
        s2orc = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc",
                             headers={'x-api-key': S2_API_KEY}).json()

        # Download the first part of the dataset
        urllib.request.urlretrieve(s2orc['files'][part],
                                   get_zip_file_for_part(part))

    # extract to file
    import gzip
    import shutil

    for i in [0]:  # range(0, 30):
        if not os.path.exists(get_unzipped_file_for_part(i)):
            zipfile = get_zip_file_for_part(i)
            with gzip.open(zipfile, 'rb') as f_in:
                with open(get_unzipped_file_for_part(i), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def get_relevant_papers_from_download(query: List[str]):
    paper_df = pd.DataFrame()
    relevant_abstracts = {}
    relevant_text = {}

    # currently just for data 0
    with open(get_unzipped_file_for_part(0), 'r') as infile:
        for line in infile:
            paper = json.loads(line)
            text = paper['content']['text']
            annotations = {k: json.loads(v) for k, v in paper['content']['annotations'].items() if v}

            def text_of(type):
                types = annotations.get(type, '')
                return [text[int(a['start']):int(a['end'])] for a in types]

            if text is not None:
                if any(x in text for x in query):
                    corpusid = str(paper['corpusid'])
                    abstract = ' '.join(set(text_of('abstract')))
                    title = ' '.join(set(text_of('title')))
                    authors = ', '.join(set(text_of('author')))
                    try:
                        url = paper['content']['source']['oainfo']['openaccessurl']
                    except TypeError:
                        url = None

                    relevant_abstracts[corpusid] = abstract
                    relevant_text[corpusid] = text
                    try:
                        doi = paper['externalids']['doi']
                    except TypeError:
                        doi = None

                    info_df = pd.DataFrame(
                        {'corpusid': [corpusid], 'DOI': [doi],
                         'title': [title], 'authors': [authors], 'oaurl': [url],
                         'abstract_path': [os.path.join(abstracts_path, corpusid + '.txt')],
                         'text_path': [os.path.join(text_path, corpusid + '.txt')]})
                    paper_df = pd.concat([paper_df, info_df])

    for c_id in relevant_abstracts:
        abstract = relevant_abstracts[c_id]
        if abstract is not None:
            f = open(os.path.join(abstracts_path, c_id + '.txt'), 'w')
            f.write(abstract)
            f.close()

    for c_id in relevant_text:
        te = relevant_text[c_id]
        if te is not None:
            f = open(os.path.join(text_path, c_id + '.txt'), 'w')
            f.write(te)
            f.close()

    paper_df.to_csv(os.path.join(paper_info_path, '_'.join(query) + '.csv'))


if __name__ == '__main__':
    # # look_at_sample()
    # get_relevant_papers_from_download("samples/s2orc/s2orc-sample.jsonl")
    # Download then check they're all there
    # TODO: find number of datasets. Slightly confused by this
    # download_fullset()

    get_relevant_papers_from_download(query=search_filtering_terms)
