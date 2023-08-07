import json
import os

import dotenv
import pandas as pd
from literature_downloads.filter_terms import is_relevant_text, query_name
from tqdm import tqdm

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
                        # See https://stackoverflow.com/questions/41728329/how-to-find-what-matched-in-any-with-python
                        matched_text = is_relevant_text(text)
                        if matched_text is not None:
                            corpusid = paper['coreId']

                            relevant_abstracts[corpusid] = paper['abstract']
                            relevant_text[corpusid] = text
                            language = paper['lan']

                            info_df = pd.DataFrame(
                                {'corpusid': [corpusid], 'DOI': [paper['doi']], 'language': [language],
                                 'matched_text': matched_text,
                                 'title': [paper['title']], 'authors': [str(paper['authors'])],
                                 'oaurl': [paper['downloadUrl']],
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
    download_full_core_dataset()
    # get_relevant_papers_from_download()
