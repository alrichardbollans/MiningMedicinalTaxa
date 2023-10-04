import json
import os
import sys

import pandas as pd

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords, sort_final_dataframe, build_output_dict

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

    import gzip
    from tqdm import tqdm
    # TODO: Add thread pool with 32?
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
                    k_word_counts = number_of_keywords(text)

                    if any(len(k_word_counts[kword_type].keys()) > 0 for kword_type in k_word_counts):

                        corpusid = str(paper['corpusid'])
                        abstract = ' '.join(set(text_of('abstract')))

                        f = open(os.path.join(sem_schol_text_path, corpusid + '.txt'), 'w')
                        f.write(text)
                        f.close()

                        if abstract != '':
                            f = open(os.path.join(sem_schol_abstracts_path, corpusid + '.txt'), 'w')
                            f.write(abstract)
                            f.close()

                        title = ' '.join(set(text_of('title')))
                        authors = ', '.join(set(text_of('author')))
                        try:
                            url = paper['content']['source']['oainfo']['openaccessurl']
                        except TypeError:
                            url = None

                        try:
                            doi = paper['externalids']['doi']
                        except TypeError:
                            doi = None

                        info_df = pd.DataFrame(build_output_dict(corpusid, doi, k_word_counts, title, authors,
                                                                 url, _rel_abstract_path, _rel_text_path))

                        paper_df = pd.concat([paper_df, info_df])

        out_df = sort_final_dataframe(paper_df)
        out_df.to_csv(os.path.join(sem_schol_paper_info_path, query_name + '.csv'))
    return out_df


def check_for_repetitions():
    'corpusid'
    paper_df = pd.read_csv(os.path.join(sem_schol_paper_info_path, query_name + '.csv'))
    problems = paper_df[paper_df.duplicated(subset=['corpusid'])]

    assert len(problems.index) == 0


if __name__ == '__main__':
    get_relevant_papers_from_download()
