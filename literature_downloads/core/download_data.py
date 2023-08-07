import json
import os

import dotenv
from typing import List

import pandas as pd
from tqdm import tqdm

from literature_downloads.filter_terms import is_relevant_text, query_name

dotenv.load_dotenv()
CORE_API_KEY = os.environ['CORE_API_KEY']
headers = {
    'x-api-key': CORE_API_KEY,
}

core_download_path = os.path.join('downloads')
core_abstracts_path = os.path.join(core_download_path, 'abstracts')
core_text_path = os.path.join(core_download_path, 'text')
core_paper_info_path = os.path.join(core_download_path, 'paper_info')
for p in [core_download_path, core_text_path, core_paper_info_path, core_abstracts_path]:
    if not os.path.exists(p):
        os.mkdir(p)


def download_full_core_dataset():
    # dataset docs = https://core.ac.uk/documentation/dataset

    os.system('wget -c https://core.ac.uk/datasets/core_2022-03-11_dataset.tar.xz')


def get_relevant_papers_from_download():
    paper_df = pd.DataFrame()
    relevant_abstracts = {}
    relevant_text = {}

    import tarfile

    zipfile = 'core_2022-03-11_dataset.tar.xz'
    print('unzipping')
    # TODO: Add thread pool
    with tarfile.open(zipfile, 'r') as main_archive:

        for member in main_archive:
            print(member)
            print(len(relevant_text))
            file_obj = main_archive.extractfile(member)
            with tarfile.open(fileobj=file_obj, mode='r') as sub_archive:
                members = sub_archive.getmembers()
                for i in tqdm(range(len(members))):
                    m = members[i]
                    if m.name.endswith('.json'):
                        f = sub_archive.extractfile(m)
                        lines = f.readlines()
                        paper = json.loads(lines[0])
                        text = paper['fullText']
                        if is_relevant_text(text):
                            corpusid = paper['coreId']

                            relevant_abstracts[corpusid] = paper['abstract']
                            relevant_text[corpusid] = text

                            info_df = pd.DataFrame(
                                {'corpusid': [corpusid], 'DOI': [paper['doi']],
                                 'title': [paper['title']], 'authors': [str(paper['authors'])], 'oaurl': [paper['downloadUrl']],
                                 'abstract_path': [os.path.join(core_abstracts_path, corpusid + '.txt')],
                                 'text_path': [os.path.join(core_text_path, corpusid + '.txt')]})
                            paper_df = pd.concat([paper_df, info_df])
    for c_id in relevant_abstracts:
        abstract = relevant_abstracts[c_id]
        if abstract is not None:
            f = open(os.path.join(core_abstracts_path, c_id + '.txt'), 'w')
            f.write(abstract)
            f.close()

    for c_id in relevant_text:
        te = relevant_text[c_id]
        if te is not None:
            f = open(os.path.join(core_text_path, c_id + '.txt'), 'w')
            f.write(te)
            f.close()

    paper_df.to_csv(os.path.join(core_paper_info_path, query_name + '.csv'))
    return paper_df


if __name__ == '__main__':
    # Getting an EOFError: Compressed file ended before the end-of-stream marker was reached
    # ./0f1/3b/223015690.json is not complete
    # wget -c https://core.ac.uk/datasets/core_2020-12-20_resync.tar.xz says file is fullly retrieved...
    download_full_core_dataset()
    # get_relevant_papers_from_download()
