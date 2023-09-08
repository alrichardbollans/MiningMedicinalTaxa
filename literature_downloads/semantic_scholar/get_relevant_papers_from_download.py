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
    relevant_abstracts = {}
    relevant_text = {}

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
                    product_kwords_dict, genusnames_dict, familynames_dict, species_dict, plantkwords_dict = number_of_keywords(text)
                    # Products
                    total_product_kword_mentions = sum(product_kwords_dict.values())
                    num_unique_product_kwords = len(product_kwords_dict.keys())
                    # Genera
                    total_genusname_mentions = sum(genusnames_dict.values())
                    num_unique_genusnames = len(genusnames_dict.keys())
                    # Families
                    total_familyname_mentions = sum(familynames_dict.values())
                    num_unique_familynames = len(familynames_dict.keys())
                    # Species
                    total_species_mentions = sum(species_dict.values())
                    unique_species_mentions = len(species_dict.keys())

                    # Plants
                    total_plantkeyword_mentions = sum(plantkwords_dict.values())
                    num_unique_plantkeywords = len(plantkwords_dict.keys())

                    if (total_product_kword_mentions > 0) or (total_genusname_mentions > 0) or (total_familyname_mentions > 0) or (
                            total_plantkeyword_mentions > 0) or (total_species_mentions > 0):

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

                        info_df = pd.DataFrame(build_output_dict(corpusid, doi, total_product_kword_mentions, num_unique_product_kwords,
                                                                 product_kwords_dict,
                                                                 total_genusname_mentions, num_unique_genusnames, genusnames_dict,
                                                                 total_familyname_mentions,
                                                                 num_unique_familynames,
                                                                 familynames_dict,
                                                                 total_species_mentions, unique_species_mentions, species_dict,
                                                                 total_plantkeyword_mentions,
                                                                 num_unique_plantkeywords, plantkwords_dict, title, authors,
                                                                 url, _rel_abstract_path, _rel_text_path))

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
