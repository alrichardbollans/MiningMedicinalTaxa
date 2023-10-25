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


def get_relevant_papers_from_download(write_texts=False):
    paper_df = pd.DataFrame()

    import tarfile

    zipfile = 'core_2022-03-11_dataset.tar.xz'
    print('unzipping')
    # TODO: Add thread pool e.g. with 32 threads for cluster
    # for train_i, test_i in kf.split(train_data_X):
    #     iter_args.append((test_i, train_data_X, train_data_y))
    # with Pool(processes=32) as pool:
    #     pool.starmap(function thats in main scope, iter_args)
    with tarfile.open(zipfile, 'r') as main_archive:
        # This is slow but useful info. # Main archive length: 10251
        # print(f'Main archive length: {len(main_archive.getnames())}')
        member_count = 1
        paper_count = 0
        for member in main_archive:
            # iterate over members then get all members out of these
            print(f'Number {member_count} of main archive containing 10251')
            member_count += 1
            print(f'Number of papers collected: {paper_count}')
            file_obj = main_archive.extractfile(member)
            with tarfile.open(fileobj=file_obj, mode='r') as sub_archive:
                # Data providers of each subarchive are here: https://core.ac.uk/data-providers
                members = sub_archive.getmembers()
                for i in tqdm(range(len(members))):
                    m = members[i]
                    if m.name.endswith('.json'):
                        f = sub_archive.extractfile(m)
                        lines = f.readlines()
                        paper = json.loads(lines[0])
                        text = paper['fullText']

                        if text is not None:
                            k_word_counts = number_of_keywords(text)
                            if any(len(k_word_counts[kword_type].keys()) > 0 for kword_type in k_word_counts):
                                paper_count += 1
                                corpusid = paper['coreId']
                                if write_texts:
                                    f = open(os.path.join(core_text_path, corpusid + '.txt'), 'w')
                                    f.write(text)
                                    f.close()

                                    if paper['abstract'] is not None:
                                        f = open(os.path.join(core_abstracts_path, corpusid + '.txt'), 'w')
                                        f.write(paper['abstract'])
                                        f.close()

                                try:
                                    language = paper['language']['code']
                                except TypeError:
                                    language = None

                                if len(paper['journals']) >= 1:
                                    journals = str(paper['journals'])
                                else:
                                    journals = None
                                if len(paper['subjects']) >= 1:
                                    subjects = str(paper['subjects'])
                                else:
                                    subjects = None
                                if len(paper['topics']) >= 1:
                                    topics = str(paper['topics'])
                                else:
                                    topics = None

                                year = paper['year']
                                issn = paper['issn']

                                info_df = pd.DataFrame(
                                    build_output_dict(corpusid, paper['doi'], year, k_word_counts, paper['title'], paper['authors'],
                                                      paper['downloadUrl'], _rel_abstract_path, _rel_text_path, language=language, journals=journals,
                                                      subjects=subjects, topics=topics, issn=issn))

                                paper_df = pd.concat([paper_df, info_df])

                out_df = sort_final_dataframe(paper_df)
                out_df.to_csv(os.path.join(core_paper_info_path, query_name + '.csv'))
    return out_df


if __name__ == '__main__':
    get_relevant_papers_from_download()
