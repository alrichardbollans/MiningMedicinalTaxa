import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords, sort_final_dataframe, build_output_dict

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
    # TODO: Add thread pool e.g.
    # for train_i, test_i in kf.split(train_data_X):
    #     iter_args.append((test_i, train_data_X, train_data_y))
    # with Pool(processes=8) as pool:
    #     pool.starmap(function thats in main scope, iter_args)
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
                            product_kwords_dict, genusnames_dict, familynames_dict, plantkwords_dict = number_of_keywords(text)
                            # Products
                            total_product_kword_mentions = sum(product_kwords_dict.values())
                            num_unique_product_kwords = len(product_kwords_dict.keys())
                            # Genera
                            total_genusname_mentions = sum(genusnames_dict.values())
                            num_unique_genusnames = len(genusnames_dict.keys())
                            # Families
                            total_familyname_mentions = sum(familynames_dict.values())
                            num_unique_familynames = len(familynames_dict.keys())
                            # Plants
                            total_plantkeyword_mentions = sum(plantkwords_dict.values())
                            num_unique_plantkeywords = len(plantkwords_dict.keys())

                            if (total_product_kword_mentions > 0) or (total_genusname_mentions > 0) or (total_familyname_mentions > 0) or (
                                    total_plantkeyword_mentions > 0):
                                corpusid = paper['coreId']

                                relevant_abstracts[corpusid] = paper['abstract']
                                relevant_text[corpusid] = text
                                try:
                                    language = paper['language']['code']
                                except TypeError:
                                    language = None

                                info_df = pd.DataFrame(
                                    build_output_dict(corpusid, paper['doi'], total_product_kword_mentions, num_unique_product_kwords,
                                                      product_kwords_dict,
                                                      total_genusname_mentions, num_unique_genusnames, genusnames_dict, total_familyname_mentions,
                                                      num_unique_familynames,
                                                      familynames_dict, total_plantkeyword_mentions,
                                                      num_unique_plantkeywords, plantkwords_dict, paper['title'], paper['authors'],
                                                      paper['downloadUrl'], _rel_abstract_path, _rel_text_path, language=language))

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
                out_df = sort_final_dataframe(paper_df)
                out_df.to_csv(os.path.join(core_paper_info_path, query_name + '.csv'))
    return out_df


if __name__ == '__main__':
    get_relevant_papers_from_download()
