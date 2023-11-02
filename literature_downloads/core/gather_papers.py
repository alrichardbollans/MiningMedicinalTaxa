import json
import os
import sys
from typing import Union, List

import pandas as pd
from tqdm import tqdm

# TODO: fix so that don't have to remake keywords
sys.path.append('..')
from core import core_paper_info_path, core_text_path, core_abstracts_path, clean_paper_text, get_info_from_core_paper


def get_papers_from_query(q_name: str, sort_order: List[str], capacity: int, out_csv: str):
    paper_info_df = pd.read_csv(os.path.join(core_paper_info_path, q_name + '.csv'), index_col=0)
    sorted_df = paper_info_df.sort_values(
        by=sort_order,
        ascending=False).reset_index(drop=True).head(capacity)

    save_texts_from_ids(sorted_df['corpusid'].values.tolist())
    sorted_df.to_csv(out_csv)


def save_texts_from_ids(ids: List[int]):
    ids_to_check = ids + [str(int_id) for int_id in ids]
    number_of_papers_to_check = len(ids)
    import tarfile

    zipfile = 'core_2022-03-11_dataset.tar.xz'
    print('unzipping')
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
                        corpusid = paper['coreId']

                        if corpusid in ids_to_check:
                            paper_count += 1
                            text = clean_paper_text(paper)
                            corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url = get_info_from_core_paper(paper)

                            f = open(os.path.join(core_text_path, corpusid + '.txt'), 'w')
                            f.write(text)
                            f.close()

                            if paper['abstract'] is not None:
                                f = open(os.path.join(core_abstracts_path, corpusid + '.txt'), 'w')
                                f.write(paper['abstract'])
                                f.close()
                            if paper_count == number_of_papers_to_check:
                                print('All papers saved')
                                return


def load_texts_from_id(given_id: Union[int, str]):
    text_path = os.path.join(core_text_path, str(given_id) + '.txt')
    text_file = open(text_path, 'r')
    # read all lines at once
    text = text_file.read()
    # close the file
    text_file.close()

    abstract_path = os.path.join(core_abstracts_path, str(given_id) + '.txt')
    abs_file = open(abstract_path, 'r')
    # read all lines at once
    abstract = abs_file.read()
    # close the file
    abs_file.close()
    return text, abstract


def example():
    kingdom_sort_order = ['plant_species_binomials_unique_total', 'fungi_species_binomials_unique_total',
                          'plant_genus_names_unique_total', 'fungi_genus_names_unique_total',
                          'plants_unique_total', 'fungi_family_names_unique_total']
    medicine_sort_order = ['medicinal entity_unique_total', 'medicinal_unique_total'] + kingdom_sort_order
    toxic_sort_order = ['toxicology entity_unique_total', 'toxicology_unique_total'] + kingdom_sort_order

    get_papers_from_query('en_medic_toxic_keywords', medicine_sort_order, 10, os.path.join(core_paper_info_path, 'top10_medicinals.csv'))
    get_papers_from_query('en_medic_toxic_keywords', toxic_sort_order, 10, os.path.join(core_paper_info_path, 'top10_toxics.csv'))


if __name__ == '__main__':
    example()
