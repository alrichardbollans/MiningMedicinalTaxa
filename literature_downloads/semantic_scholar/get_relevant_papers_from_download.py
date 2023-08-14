import json
import os
import sys

import pandas as pd

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords

from literature_downloads.semantic_scholar import get_zip_file_for_part, sem_schol_download_path

scratch_path = os.environ.get('SCRATCH')

sem_schol_abstracts_path = os.path.join(sem_schol_download_path, 'abstracts')
_rel_abstract_path = os.path.relpath(sem_schol_abstracts_path, scratch_path)
sem_schol_text_path = os.path.join(sem_schol_download_path, 'text')
_rel_text_path = os.path.relpath(sem_schol_text_path, scratch_path)

sem_schol_paper_info_path = os.path.join(sem_schol_download_path, 'paper_info')
_dirs = [sem_schol_abstracts_path, sem_schol_text_path, sem_schol_paper_info_path]
for d in _dirs:
    if not os.path.exists(d):
        os.mkdir(d)


def get_relevant_papers_from_download():
    paper_df = pd.DataFrame()
    relevant_abstracts = {}
    relevant_text = {}

    import gzip
    from tqdm import tqdm
    # TODO: Add thread pool?
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

                if text is not None:
                    kwords_dict, plantnames_dict, plantkwords_dict = number_of_keywords(text)
                    total_kword_mentions = sum(kwords_dict.values())
                    num_unique_kwords = len(kwords_dict.keys())
                    total_plantname_mentions = sum(plantnames_dict.values())
                    num_unique_plantnames = len(plantnames_dict.keys())

                    total_plantkeyword_mentions = sum(plantkwords_dict.values())
                    num_unique_plantkeywords = len(plantkwords_dict.keys())
                    if (total_kword_mentions > 0) or (total_plantname_mentions > 0) or (total_plantkeyword_mentions > 0):

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
                             'total_keyword_mentions': total_kword_mentions,
                             'unique_keyword_mentions': num_unique_kwords,
                             'keyword_count': str(kwords_dict),
                             'total_plantname_mentions': total_plantname_mentions,
                             'unique_plantname_mentions': num_unique_plantnames,
                             'plantname_count': str(plantnames_dict),
                             'total_plantkeyword_mentions': total_plantkeyword_mentions,
                             'unique_plantkeyword_mentions': num_unique_plantkeywords,

                             'plantkeyword_count': str(plantkwords_dict),
                             'title': [title], 'authors': [authors], 'oaurl': [url],
                             'abstract_path': [os.path.join(_rel_abstract_path, corpusid + '.txt')],
                             'text_path': [os.path.join(_rel_text_path, corpusid + '.txt')]})
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


if __name__ == '__main__':
    get_relevant_papers_from_download()
