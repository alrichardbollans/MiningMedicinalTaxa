import json
import multiprocessing
import os
import tarfile
import zipfile
from typing import Union, List

import pandas as pd
from tqdm import tqdm

from literature_downloads.core import core_paper_info_path, clean_paper_text, core_text_path, core_abstracts_path, \
    CORE_TAR_FILE


def get_papers_from_query(q_name: str, sort_order: List[str], capacity: int, out_csv: str, folder_tag: str):
    query_zip = os.path.join(core_paper_info_path, q_name + '.zip')  # zip downloaded from cluster
    with zipfile.ZipFile(query_zip, "r") as zf:
        df_files = [f for f in zf.namelist() if f.endswith('.csv')]

        paper_info_df = pd.concat(
            [pd.read_csv(zf.open(f)).sort_values(by=sort_order, ascending=False).reset_index(drop=True).head(capacity) for f in
             df_files])  ## Limit capacity to save memory
    sorted_df = paper_info_df.sort_values(
        by=sort_order,
        ascending=False).reset_index(drop=True).head(capacity)
    sorted_df.to_csv(out_csv)
    save_texts_from_ids(sorted_df['corpusid'].values.tolist(), sorted_df['tar_archive_name'].values.tolist(), folder_tag)


def save_texts_from_provider(provider, providers, ids_to_check, folder_tag, get_abstracts):
    paper_count_from_provider = 0
    provider_name = os.path.basename(provider.name)
    if provider_name in providers:
        with tarfile.open(CORE_TAR_FILE, 'r') as main_archive:
            provider_file_obj = main_archive.extractfile(provider)

            with tarfile.open(fileobj=provider_file_obj, mode='r') as sub_archive:
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
                            paper_count_from_provider += 1
                            text = clean_paper_text(paper)
                            corpusid = paper['coreId']

                            f = open(os.path.join(core_text_path, folder_tag, corpusid + '.txt'), 'w')
                            f.write(text)
                            f.close()

                            if get_abstracts:
                                if paper['abstract'] is not None:
                                    f = open(os.path.join(core_abstracts_path, folder_tag, corpusid + '.txt'), 'w')
                                    f.write(paper['abstract'])
                                    f.close()
        print(f'{paper_count_from_provider} papers collected from: {provider_name}')

    return paper_count_from_provider


def save_texts_from_ids(ids: List[int], providers: List[int], folder_tag: str, get_abstracts: bool = False):
    ids_to_check = ids + [str(int_id) for int_id in ids]
    number_of_papers_to_check = len(ids)

    if not os.path.exists(os.path.join(core_text_path, folder_tag)):
        os.mkdir(os.path.join(core_text_path, folder_tag))
    if not os.path.exists(os.path.join(core_abstracts_path, folder_tag)):
        os.mkdir(os.path.join(core_abstracts_path, folder_tag))

    # TODO: Need to return values to get paper counts
    with tarfile.open(CORE_TAR_FILE, 'r') as main_archive:
        with multiprocessing.Pool(6) as pool:
            tasks = [pool.apply_async(save_texts_from_provider, args=(member, providers, ids_to_check, folder_tag, get_abstracts,)) for member in
                     main_archive]

            # Wait for all tasks to complete
            for task in tasks:
                task.get()


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

    get_papers_from_query('en_medic_toxic_keywords', medicine_sort_order, 1, os.path.join(core_text_path, 'top50_medicinals.csv'), 'medicinal')
    get_papers_from_query('en_medic_toxic_keywords', toxic_sort_order, 1, os.path.join(core_text_path, 'top50_toxics.csv'), 'toxic')


if __name__ == '__main__':
    example()
