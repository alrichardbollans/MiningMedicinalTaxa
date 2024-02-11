import ast
import fnmatch
import json
import os
import zipfile
from typing import Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from literature_downloads.core import core_paper_info_path, core_download_path

KINGDOM_SORT_ORDER = ['plant_species_binomials_unique_total', 'fungi_species_binomials_unique_total',
                      'plant_genus_names_unique_total', 'fungi_genus_names_unique_total',
                      'plant_family_names_unique_total', 'fungi_family_names_unique_total']
medicine_sort_order = ['medicinal entity_unique_total', 'medicinal_unique_total'] + KINGDOM_SORT_ORDER
toxic_sort_order = ['toxicology entity_unique_total', 'toxicology_unique_total'] + KINGDOM_SORT_ORDER


class CQuery:
    def __init__(self, zip_file: str, output_dir: str, extracted_paper_csv_name: str, sort_order: List[str], capacity: int):
        self.zip_file = os.path.join(core_paper_info_path, zip_file)

        self.output_dir = output_dir + f'_top_{str(capacity)}'
        self.summary_csv = os.path.join(self.output_dir, zip_file + '_summary.csv')  # This is only dependent on the zip
        self.extracted_paper_csv = os.path.join(self.output_dir, extracted_paper_csv_name.replace('.csv', f'_top_{str(capacity)}.csv'))
        self.sort_order = sort_order
        self.capacity = capacity
        self.tmp_dl_path = os.path.join(self.output_dir, 'tmp_downloads_for_core_providers_salt1982733')
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

            elif df.columns.tolist() != second_columns:
                print(df)
                raise ValueError(f'{df.columns.tolist()}')

        else:

            # print(f'Skipping {name}: Columns {df.columns.tolist()}')
            assert len(df.index) == 0
            assert df.columns.tolist() == ['corpusid', 'tar_archive_name']
            return None
        return df

    @staticmethod
    def get_papers_with_value_greater_than_zero(df: pd.DataFrame, colstocheck=None) -> pd.DataFrame:
        if colstocheck is None:
            colstocheck = KINGDOM_SORT_ORDER
        mask = np.column_stack([df[col] > 0 for col in colstocheck])
        df = df.loc[mask.any(axis=1)]
        return df

    def sort_df(self, df: pd.DataFrame):
        # Only return papers with Kingdom mentions
        df = self.get_papers_with_value_greater_than_zero(df)
        # Sort by sort order at get top
        df = df.sort_values(by=self.sort_order, ascending=False).reset_index(drop=True).head(
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
                # if len(df.columns) > 2:
                #     df = self.sort_df(df)
                df = self.clean_paper_df(df)
                if df is not None:
                    paper_df = pd.concat([paper_df, df])
                # Periodically sort dataframe to save memory
                # if i % 2000 == 0 and i != 0:
                #     paper_df = self.sort_df(paper_df)
        # Summarise
        print('summarising paper dataframe...')
        paper_df[['corpusid', 'DOI', 'year', 'language', 'tar_archive_name', 'fungi_family_names_counts', 'plant_family_names_counts']].describe(
            include='all').to_csv(
            self.summary_csv)

        print('final sort')
        paper_df = self.sort_df(paper_df)
        print('writing')
        paper_df.to_csv(self.extracted_paper_csv)

    def download_providers(self):
        paper_df = pd.read_csv(self.extracted_paper_csv)
        providers = paper_df['tar_archive_name'].unique()
        print(providers)

        cluster_directory_path = 'arichard@gruffalo.cropdiversity.ac.uk:/mnt/shared/scratch/arichard/MedicinalPlantMining/literature_downloads/core/extracted_core/data-ext/resync/output/tmp/'
        base_command = f'scp -r {cluster_directory_path}'
        if not os.path.exists(self.tmp_dl_path):
            os.mkdir(self.tmp_dl_path)
        for provider in providers:
            new_command = base_command + f'{provider.replace(".tar.xz", "")} {self.tmp_dl_path}'
            os.system('pwd')
            os.system(new_command)

    def save_texts_from_provider(self, provider):
        if 'tar.xz' not in provider:
            raise ValueError()
        if not os.path.exists(self.text_dump_path):
            os.mkdir(self.text_dump_path)
        paper_df = pd.read_csv(self.extracted_paper_csv)
        provider_df = paper_df[paper_df['tar_archive_name'] == provider]

        relevant_ids = provider_df['corpusid'].astype('string').values

        # find all .json files recursively in nested directories in PATH
        json_files = {}
        for root, dirnames, filenames in os.walk(self.tmp_dl_path):
            for filename in fnmatch.filter(filenames, '*.json'):
                corpus_id = filename.replace('.json', '')
                if corpus_id in relevant_ids:
                    filepath = os.path.join(root, filename)
                    json_files[corpus_id] = filepath
                    with open(filepath, 'r') as infile:
                        json_data = json.load(infile)['fullText']
                    with open(os.path.join(self.text_dump_path, corpus_id+'.txt'), "w") as text_file:
                        text_file.write(json_data)
        raise ValueError('add some checks')
    def load_texts_from_id(self, given_id: Union[int, str]):
        pass


if __name__ == '__main__':
    medicinal_query = CQuery('en_medic_toxic_keywords3.zip', os.path.join(core_download_path, 'medicinals'),
                             'medicinals.csv',
                             medicine_sort_order,
                             10000)

    medicinal_query.extract_query_zip()

    toxic_query = CQuery('en_medic_toxic_keywords3.zip', os.path.join(core_download_path, 'toxics.csv'),
                         'toxics.csv',
                         toxic_sort_order, 10000)
    toxic_query.extract_query_zip()
