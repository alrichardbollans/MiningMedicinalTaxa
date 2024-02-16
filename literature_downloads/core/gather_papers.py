import ast
import fnmatch
import json
import os
import time
import zipfile
from typing import Union, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from literature_downloads.core import core_paper_info_path, core_download_path, extracted_core_path
import dotenv

dotenv.load_dotenv()
CROP_DATA_SSH_PATH = os.environ['CROP_DATA_SSH_PATH']
CORE_API_KEY = os.environ['CORE_API_KEY']

KINGDOM_SORT_ORDER = ['plant_species_binomials_unique_total', 'fungi_species_binomials_unique_total',
                      'plant_genus_names_unique_total', 'fungi_genus_names_unique_total',
                      'plant_family_names_unique_total', 'fungi_family_names_unique_total']
medicine_sort_order = ['medicinal entity_unique_total', 'medicinal_unique_total']
toxic_sort_order = ['toxicology entity_unique_total', 'toxicology_unique_total']


class CQuery:
    def __init__(self, zip_file: str, output_dir: str, extracted_paper_csv_name: str, sort_order: List[str], capacity: int):
        self.zip_file = os.path.join(core_paper_info_path, zip_file)

        self.output_dir = output_dir + f'_top_{str(capacity)}'
        self.summary_csv = os.path.join(self.output_dir, zip_file + '_summary.csv')  # This is only dependent on the zip
        self.extracted_paper_csv = os.path.join(self.output_dir, extracted_paper_csv_name.replace('.csv', f'_top_{str(capacity)}.csv'))
        self.extracted_paper_summary_csv = self.extracted_paper_csv.replace('.csv', f'_summary.csv')
        self.sort_order = sort_order
        self.capacity = capacity
        self.tmp_dl_path = os.path.join(extracted_core_path, 'misc_jsons')
        self.text_dump_path = os.path.join(self.output_dir, 'texts')

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    @staticmethod
    def clean_paper_df(df: pd.DataFrame) -> pd.DataFrame:
        # This is currently messy but can be tidied for future versions
        # It is also slow because of literal_eval
        # Clean to match newer versions with previous versions of get_rel script
        # Apply some checks
        def optimised_dict_conversion(input_str: str):
            if input_str == '{}':
                out = np.nan
            else:
                out = list(ast.literal_eval(input_str).keys())
            return out

        first_columns = ['corpusid', 'oai', 'DOI', 'year', 'language', 'journals', 'issn', 'subjects', 'topics', 'title', 'authors',
                         'oaurl', 'abstract_path', 'text_path', 'fungi_genus_names_counts', 'fungi_genus_names_total',
                         'fungi_genus_names_unique_total', 'fungi_species_binomials_counts', 'fungi_species_binomials_total',
                         'fungi_species_binomials_unique_total', 'fungi_counts', 'fungi_total', 'fungi_unique_total',
                         'fungi_family_names_counts', 'fungi_family_names_total', 'fungi_family_names_unique_total',
                         'plant_family_names_counts', 'plant_family_names_total', 'plant_family_names_unique_total',
                         'toxicology_counts', 'toxicology_total', 'toxicology_unique_total', 'medicinal_counts', 'medicinal_total',
                         'medicinal_unique_total', 'medicinal entity_counts', 'medicinal entity_total',
                         'medicinal entity_unique_total', 'plant_species_binomials_counts', 'plant_species_binomials_total',
                         'plant_species_binomials_unique_total', 'toxicology entity_counts', 'toxicology entity_total',
                         'toxicology entity_unique_total', 'plant_genus_names_counts', 'plant_genus_names_total',
                         'plant_genus_names_unique_total', 'lifeform_counts', 'lifeform_total', 'lifeform_unique_total',
                         'plants_counts', 'plants_total', 'plants_unique_total', 'tar_archive_name']
        second_columns = ['corpusid', 'oai', 'DOI', 'year', 'language', 'journals', 'issn', 'subjects', 'topics', 'title', 'authors', 'oaurl',
                          'abstract_path', 'text_path', 'fungi_genus_names_counts', 'fungi_genus_names_unique_total',
                          'fungi_species_binomials_counts',
                          'fungi_species_binomials_unique_total', 'fungi_counts', 'fungi_unique_total', 'fungi_family_names_counts',
                          'fungi_family_names_unique_total', 'plant_family_names_counts', 'plant_family_names_unique_total', 'toxicology_counts',
                          'toxicology_unique_total', 'medicinal_counts', 'medicinal_unique_total', 'medicinal entity_counts',
                          'medicinal entity_unique_total', 'plant_species_binomials_counts', 'plant_species_binomials_unique_total',
                          'toxicology entity_counts', 'toxicology entity_unique_total', 'plant_genus_names_counts', 'plant_genus_names_unique_total',
                          'lifeform_counts', 'lifeform_unique_total', 'plants_counts', 'plants_unique_total', 'tar_archive_name']
        # These should be dropped
        in_old_but_not_new = ['fungi_genus_names_total', 'fungi_species_binomials_total', 'fungi_total', 'fungi_family_names_total',
                              'plant_family_names_total', 'toxicology_total', 'medicinal_total', 'medicinal entity_total',
                              'plant_species_binomials_total', 'toxicology entity_total', 'plant_genus_names_total', 'lifeform_total', 'plants_total']
        if len(df.columns) > 2:

            if df.columns.tolist() == first_columns:
                for c in in_old_but_not_new:
                    unq_c = c.replace('_total', '_unique_total')
                    problems = df[df[unq_c].gt(df[c])]
                    assert len(problems.index) == 0
                df = df.drop(columns=in_old_but_not_new)
                for col in df.columns:
                    if 'count' in col:
                        df[col] = df[col].apply(optimised_dict_conversion)

            elif set(df.columns.tolist()) != set(second_columns):
                print(df)
                raise ValueError(f'{df.columns.tolist()}')

        else:

            # print(f'Skipping {name}: Columns {df.columns.tolist()}')
            assert len(df.index) == 0
            assert df.columns.tolist() == ['corpusid', 'tar_archive_name']
            return None
        return df

    @staticmethod
    def get_papers_with_value_greater_than_zero(df: pd.DataFrame, colstocheck: List[str]) -> pd.DataFrame:

        mask = np.column_stack([df[col] > 0 for col in colstocheck])
        df = df.loc[mask.any(axis=1)]
        return df

    def sort_df(self, df: pd.DataFrame):
        # Only return papers with Kingdom mentions
        df = self.get_papers_with_value_greater_than_zero(df, KINGDOM_SORT_ORDER)
        # Only return papers with mentions from sort order
        df = self.get_papers_with_value_greater_than_zero(df, self.sort_order)
        # Sort by sort order and then kingdom order and then get top
        df = df.sort_values(by=self.sort_order + KINGDOM_SORT_ORDER, ascending=False).reset_index(drop=True).head(
            self.capacity)
        return df

    def extract_query_zip(self):
        # Extract filtered papers into one clean dataframe
        paper_df = pd.DataFrame()
        with zipfile.ZipFile(self.zip_file, "r") as zf:
            namelist = zf.namelist()
            df_files = [f for f in namelist if f.endswith('.csv')]

            for i in tqdm(range(len(df_files))):
                name = df_files[i]
                df = pd.read_csv(zf.open(name))
                if len(df.columns) > 2:
                    ## Sorting is faster than cleaning
                    df = self.sort_df(df)
                df = self.clean_paper_df(df)
                if df is not None:
                    paper_df = pd.concat([paper_df, df])
                # Periodically sort dataframe to save memory
                if i % 2000 == 0 and i != 0:
                    paper_df = self.sort_df(paper_df)

        print('final sort')
        paper_df = self.sort_df(paper_df)
        print('writing')
        paper_df.to_csv(self.extracted_paper_csv)
        self.summarise_papers()

    def summarise_papers(self):
        paper_df = pd.read_csv(self.extracted_paper_csv)
        assert len(paper_df.index) == self.capacity
        print('summarising')
        paper_df[['corpusid', 'DOI', 'year', 'language', 'tar_archive_name', 'fungi_family_names_counts', 'plant_family_names_counts']].describe(
            include='all').to_csv(self.extracted_paper_summary_csv)

    def summarise_zip(self):
        ''' A summary that doesn't require any cleaning'''
        cols_to_summarise = ['corpusid', 'DOI', 'year', 'language', 'tar_archive_name']
        paper_df = pd.DataFrame()
        with zipfile.ZipFile(self.zip_file, "r") as zf:
            namelist = zf.namelist()
            df_files = [f for f in namelist if f.endswith('.csv')]

            for i in tqdm(range(len(df_files))):
                name = df_files[i]
                df = pd.read_csv(zf.open(name))
                if len(df.columns) > 2:
                    paper_df = pd.concat([paper_df, df[cols_to_summarise]])

        # Summarise
        print('summarising paper dataframe...')
        paper_df.describe(
            include='all').to_csv(
            self.summary_csv)

    def download_providers(self):

        paper_df = pd.read_csv(self.extracted_paper_csv)
        providers = paper_df['tar_archive_name'].unique()
        print(providers)

        cluster_directory_path = CROP_DATA_SSH_PATH + '/extracted_core/data-ext/resync/output/tmp/'
        base_command = f'rsync --archive --progress --ignore-existing {cluster_directory_path}'
        if not os.path.exists(extracted_core_path):
            os.mkdir(extracted_core_path)
        for provider in providers:
            new_command = base_command + f'{provider.replace(".tar.xz", "")} {extracted_core_path}'
            os.system('pwd')
            os.system(new_command)

    @staticmethod
    def download_json_from_coreid(coreid: str, outpath: str):
        if not os.path.isfile(outpath):
            # print(f'Downloading {coreid} to {outpath}')
            time.sleep(5)  # 10,000 tokens per day, maximum 10 per minute.
            headers = {"Authorization": "Bearer " + CORE_API_KEY}
            response = requests.get(f"https://api.core.ac.uk/v3/outputs/{coreid}", headers=headers)
            output_json = response.json()
            if response.status_code != 200:
                print(output_json)
                print(f'Response code: {response.status_code} for {coreid}')
            else:
                with open(outpath, 'w') as f:
                    json.dump(output_json, f)

                return output_json
        else:
            # print(f'Already downloaded: {coreid} at {outpath}')
            with open(outpath, 'r') as infile:
                json_data = json.load(infile)
            return json_data

    def save_texts_from_provider(self, provider):
        if 'tar.xz' not in provider:
            raise ValueError()
        if not os.path.exists(self.text_dump_path):
            os.mkdir(self.text_dump_path)
        if not os.path.exists(self.tmp_dl_path):
            os.mkdir(self.tmp_dl_path)
        paper_df = pd.read_csv(self.extracted_paper_csv)
        provider_df = paper_df[paper_df['tar_archive_name'] == provider]

        relevant_ids = provider_df['corpusid'].astype('string').values

        not_worked = []
        provider_dir = provider.replace(".tar.xz", "")
        provider_download_path = os.path.join(self.tmp_dl_path, provider_dir)
        if not os.path.exists(provider_download_path):
            os.mkdir(provider_download_path)
        for corpus_id in relevant_ids:
            record_path = os.path.join(provider_download_path, corpus_id + '.json')

            returned_json = self.download_json_from_coreid(corpus_id, record_path)
            try:
                assert str(returned_json['dataProvider']['id']) == provider_dir
                fulltext = returned_json['fullText']
                with open(os.path.join(self.text_dump_path, corpus_id + '.txt'), "w") as text_file:
                    text_file.write(fulltext)
            except (AssertionError, TypeError):
                not_worked.append(corpus_id)
                print(f'not worked to manually get: {not_worked}')

    def save_all_texts(self):
        paper_df = pd.read_csv(self.extracted_paper_csv)
        providers = paper_df['tar_archive_name'].unique().tolist()
        for p in tqdm(providers):
            self.save_texts_from_provider(p)

    def load_texts_from_id(self, given_id: Union[int, str]):
        pass


if __name__ == '__main__':
    medicinal_query = CQuery('en_medic_toxic_keywords_final.zip', os.path.join(core_download_path, 'medicinals'),
                             'medicinals.csv',
                             medicine_sort_order,
                             10000)
    # medicinal_query.summarise_zip()
    # medicinal_query.extract_query_zip()
    medicinal_query.save_all_texts()

    toxic_query = CQuery('en_medic_toxic_keywords_final.zip', os.path.join(core_download_path, 'toxics'),
                         'toxics.csv',
                         toxic_sort_order, 10000)
    # toxic_query.summarise_zip()
    # toxic_query.extract_query_zip()
    toxic_query.save_all_texts()
