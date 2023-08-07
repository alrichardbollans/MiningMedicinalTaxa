import json
import os.path
import time
from typing import List

import pandas as pd

import dotenv

from literature_downloads.filter_terms import is_relevant_text, query_name

dotenv.load_dotenv()
S2_API_KEY = os.environ['S2_API_KEY']
headers = {
    'x-api-key': S2_API_KEY,
}

# docs: https://api.semanticscholar.org/api-docs/datasets

sem_schol_download_path = os.path.join('downloads')
sem_schol_dataset_download_path = os.path.join(sem_schol_download_path, 'datasets')
sem_schol_abstracts_path = os.path.join(sem_schol_download_path, 'abstracts')
sem_schol_text_path = os.path.join(sem_schol_download_path, 'text')
sem_schol_paper_info_path = os.path.join(sem_schol_download_path, 'paper_info')
if not os.path.exists(sem_schol_dataset_download_path):
    os.mkdir(sem_schol_dataset_download_path)


def get_zip_file_for_part(part: int):
    return os.path.join(sem_schol_dataset_download_path, "s2orc-part" + str(part) + ".jsonl.gz")


def get_unzipped_file_for_part(part: int):
    return os.path.join(sem_schol_dataset_download_path, "s2orc-part" + str(part) + ".jsonl")


def download_fullset():
    import requests
    import urllib
    # Get info about the latest release
    latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
    print(latest_release['README'])
    print(latest_release['release_id'])

    # Print names of datasets in the release
    print("\n".join(d['name'] for d in latest_release['datasets']))

    # Print README for one of the datasets
    print(latest_release['datasets'][2]['README'])

    for part in range(0, 30):
        # Get info about the s2orc dataset
        s2orc = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc",
                             headers={'x-api-key': S2_API_KEY}).json()
        print(part)
        # Download the part of the dataset
        urllib.request.urlretrieve(s2orc['files'][part],
                                   get_zip_file_for_part(part))
        print(f'part {part} done')
        time.sleep(5)

    # extract to file
    # import gzip
    # import shutil
    #
    # for i in [0]:  # range(0, 30):
    #     if not os.path.exists(get_unzipped_file_for_part(i)):
    #         zipfile = get_zip_file_for_part(i)
    #         with gzip.open(zipfile, 'rb') as f_in:
    #             with open(get_unzipped_file_for_part(i), 'wb') as f_out:
    #                 shutil.copyfileobj(f_in, f_out)


def get_relevant_papers_from_download():
    paper_df = pd.DataFrame()
    relevant_abstracts = {}
    relevant_text = {}

    import gzip
    from tqdm import tqdm
    # TODO: Add thread pool?
    # TODO: Add found term
    # Simple query takes 6 mins per part
    for part in tqdm(range(0, 30)):
        zipfile = get_zip_file_for_part(part)
        print('unzipping')
        with gzip.open(zipfile, 'rb') as infile:
            print('unzipped')
            for line in infile:
                paper = json.loads(line)
                text = paper['content']['text']
                annotations = {k: json.loads(v) for k, v in paper['content']['annotations'].items() if v}

                def text_of(type):
                    types = annotations.get(type, '')
                    return [text[int(a['start']):int(a['end'])] for a in types]

                if is_relevant_text(text):

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
                         'abstract_path': [os.path.join(sem_schol_abstracts_path, corpusid + '.txt')],
                         'text_path': [os.path.join(sem_schol_text_path, corpusid + '.txt')]})
                    paper_df = pd.concat([paper_df, info_df])

        for c_id in relevant_abstracts:
            abstract = relevant_abstracts[c_id]
            if abstract is not None:
                f = open(os.path.join(sem_schol_abstracts_path, c_id + '.txt'), 'w')
                f.write(abstract)
                f.close()

        for c_id in relevant_text:
            te = relevant_text[c_id]
            if te is not None:
                f = open(os.path.join(sem_schol_text_path, c_id + '.txt'), 'w')
                f.write(te)
                f.close()

        paper_df.to_csv(os.path.join(sem_schol_paper_info_path, query_name + '.csv'))
    return paper_df


def check_for_repetitions():
    'corpusid'
    paper_df = pd.read_csv(os.path.join(sem_schol_paper_info_path, query_name + '.csv'))
    problems = paper_df[paper_df.duplicated(subset=['corpusid'])]

    assert len(problems.index) == 0


def check_for_subsumption_by_core():
    # Check what's already in core and remove rest to save space, also provide a statistic for this. Actually think this will be really expensive.
    pass


if __name__ == '__main__':
    # Download then check they're all there
    # download_fullset()
    get_relevant_papers_from_download()
    check_for_repetitions()
    check_for_subsumption_by_core()
