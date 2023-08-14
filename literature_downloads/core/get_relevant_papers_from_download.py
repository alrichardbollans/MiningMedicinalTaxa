import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords

scratch_path = os.environ.get('SCRATCH')

core_project_path = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'core')
core_download_path = os.path.join(core_project_path, 'downloads')
core_abstracts_path = os.path.join(core_download_path, 'abstracts')
_rel_abstract_path = os.path.relpath(core_abstracts_path, scratch_path)

core_text_path = os.path.join(core_download_path, 'text')
_rel_text_path = os.path.relpath(core_text_path, scratch_path)

core_paper_info_path = os.path.join(core_download_path, 'paper_info')
for p in [core_download_path, core_text_path, core_paper_info_path, core_abstracts_path]:
    if not os.path.exists(p):
        os.mkdir(p)


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
            # iterate over members then get all members out of these
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

                        if text is not None:
                            kwords_dict, plantnames_dict, plantkwords_dict = number_of_keywords(text)
                            total_kword_mentions = sum(kwords_dict.values())
                            num_unique_kwords = len(kwords_dict.keys())
                            total_plantname_mentions = sum(plantnames_dict.values())
                            num_unique_plantnames = len(plantnames_dict.keys())

                            total_plantkeyword_mentions = sum(plantkwords_dict.values())
                            num_unique_plantkeywords = len(plantkwords_dict.keys())

                            if (total_kword_mentions > 0) or (total_plantname_mentions > 0) or (total_plantkeyword_mentions > 0):
                                corpusid = paper['coreId']

                                relevant_abstracts[corpusid] = paper['abstract']
                                relevant_text[corpusid] = text
                                try:
                                    language = paper['language']['code']
                                except TypeError:
                                    language = None

                                info_df = pd.DataFrame(
                                    {'corpusid': [corpusid], 'DOI': [paper['doi']], 'language': [language],
                                     'total_keyword_mentions': total_kword_mentions,
                                     'unique_keyword_mentions': num_unique_kwords,
                                     'keyword_count': str(kwords_dict),
                                     'total_plantname_mentions': total_plantname_mentions,
                                     'unique_plantname_mentions': num_unique_plantnames,
                                     'plantname_count': str(plantnames_dict),
                                     'total_plantkeyword_mentions': total_plantkeyword_mentions,
                                     'unique_plantkeyword_mentions': num_unique_plantkeywords,

                                     'plantkeyword_count': str(plantkwords_dict),
                                     'title': [paper['title']], 'authors': [str(paper['authors'])],
                                     'oaurl': [paper['downloadUrl']],
                                     'abstract_path': [os.path.join(_rel_abstract_path, corpusid + '.txt')],
                                     'text_path': [os.path.join(_rel_text_path, corpusid + '.txt')]})
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
    get_relevant_papers_from_download()
