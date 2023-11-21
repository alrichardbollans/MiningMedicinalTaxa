import json
import multiprocessing
import os
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List

import pandas as pd
from tqdm import tqdm

from literature_downloads.core import core_paper_info_path, clean_paper_text, core_text_path, core_abstracts_path, \
    CORE_TAR_FILE, core_download_path


def get_papers_from_query(q_name: str, sort_order: List[str], capacity: int, out_csv: str, folder_tag: str):
    query_zip = os.path.join(core_paper_info_path, q_name + '.zip')  # zip downloaded from cluster
    with zipfile.ZipFile(query_zip, "r") as zf:
        namelist = zf.namelist()
        df_files = [f for f in namelist if f.endswith('.csv')]
        dfs = []
        for i in tqdm(range(len(df_files))):
            dfs.append(pd.read_csv(zf.open(df_files[i])).sort_values(by=sort_order, ascending=False).reset_index(drop=True).head(
                capacity))  ## Limit capacity to save memory
        paper_info_df = pd.concat(dfs)
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

                            with open(os.path.join(core_text_path, folder_tag, corpusid + '.txt'), 'w') as textfile:
                                textfile.write(text)

                            if get_abstracts:
                                if paper['abstract'] is not None:
                                    with open(os.path.join(core_abstracts_path, folder_tag, corpusid + '.txt'), 'w') as absfile:
                                        absfile.write(paper['abstract'])
    print(f'{paper_count_from_provider} papers collected from: {provider_name}')

    return paper_count_from_provider


def save_texts_from_ids(ids: List[int], providers: List[int], folder_tag: str, get_abstracts: bool = False):
    ids_to_check = ids + [str(int_id) for int_id in ids]
    number_of_papers_to_check = len(ids)

    if not os.path.exists(os.path.join(core_text_path, folder_tag)):
        os.mkdir(os.path.join(core_text_path, folder_tag))
    if not os.path.exists(os.path.join(core_abstracts_path, folder_tag)):
        os.mkdir(os.path.join(core_abstracts_path, folder_tag))
    BATCH_SIZE = 1000
    with tarfile.open(CORE_TAR_FILE, 'r') as main_archive:
        with multiprocessing.Pool(64) as pool:
            i = 0
            for member in main_archive:
                fileCounter = 0
                for root, dirs, files in os.walk(os.path.join(core_text_path, folder_tag)):
                    for file in files:
                        if file.endswith('.txt'):
                            fileCounter += 1
                if fileCounter == number_of_papers_to_check:
                    tasks = []
                    print('Collected all papers')
                    break
                else:
                    if i % BATCH_SIZE == 0:
                        tasks = []
                    tasks.append(pool.apply_async(save_texts_from_provider, args=(member, providers, ids_to_check, folder_tag, get_abstracts,)))

                    if i % BATCH_SIZE == BATCH_SIZE - 1:
                        for task in tasks:
                            task.get()
                        tasks = []
                    i += 1

            # Remaining partial batch
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

    get_papers_from_query('en_medic_toxic_keywords', medicine_sort_order, 10, os.path.join(core_download_path, 'top10_medicinals.csv'),
                          'medicinal')
    get_papers_from_query('en_medic_toxic_keywords', toxic_sort_order, 10, os.path.join(core_download_path, 'top10_toxics.csv'), 'toxic')


if __name__ == '__main__':
    example()
