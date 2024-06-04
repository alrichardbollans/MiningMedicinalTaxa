import fnmatch
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords, build_output_dict
from useful_string_methods import remove_unneccesary_lines

scratch_path = os.environ.get('KEWSCRATCHPATH')
conda_path = os.environ.get('KEWDATAPATH') # A separate path for a non networked drive to handle a large number of files and directories

core_project_path = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'core')
core_download_path = os.path.join(core_project_path, 'downloads')
extracted_core_path = os.path.join(conda_path, 'extracted_core', 'data-ext', 'resync', 'output', 'tmp')

# TODO: Remove abstract and text paths in future versions. Instead save text path in extracted_core so that its easy to retrieve.
core_abstracts_path = os.path.join(core_download_path, 'abstracts')
_rel_abstract_path = os.path.relpath(core_abstracts_path, scratch_path)

core_text_path = os.path.join(core_download_path, 'text')
_rel_text_path = os.path.relpath(core_text_path, scratch_path)

core_paper_info_path = os.path.join(core_download_path, 'paper_info')
core_paper_info_query_path = os.path.join(core_paper_info_path, query_name)
for p in [core_download_path, core_paper_info_path, core_paper_info_query_path]:
    if not os.path.exists(p):
        os.mkdir(p)

CORE_TAR_FILE = 'core_2022-03-11_dataset.tar.xz'

# Compile regexes once first
# Split by looking for an instance of simple_string (ignoring case) begins a line on its own (or with line numbers) followed by any amount of whitespace and then a new line
# Must use re.MULTILINE flag such that the pattern character '^' matches at the beginning of the string and at the beginning of each line (immediately following each newline)
_reference_regex = re.compile(r"^\s*\d*\s*References\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_supp_regex = re.compile(r"^\s*\d*\s*Supplementary material\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_conf_regex = re.compile(r"^\s*\d*\s*Conflict of interest\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_ackno_regex = re.compile(r"^\s*\d*\s*Acknowledgments\s*\n", flags=re.IGNORECASE | re.MULTILINE)


def clean_title_strings(given_title: str) -> str:
    ## Also need to fix encodings e.g. \\u27
    if given_title is not None:
        fixed_whitespace = ' '.join(given_title.split())
        return fixed_whitespace
    else:
        return given_title


def retrieve_text_before_phrase(given_text: str, my_regex, simple_string: str) -> str:
    if simple_string.lower() in given_text.lower():

        text_split = my_regex.split(given_text, maxsplit=1)  # This is the bottleneck

        pre_split = text_split[0]
        if len(text_split) > 1:
            # At most 1 split occurs, if there has been a split the remainder of the string is returned as the final element of the list.
            # if text after split point is longer than before the split, then revert to given text.
            post_split = text_split[1]
            if len(post_split) > len(pre_split):
                pre_split = given_text

        return pre_split
    else:

        return given_text


def clean_paper_text(paper: dict) -> str:
    text = paper['fullText']
    if text is None:
        return None
    given_text_cleaned = remove_unneccesary_lines(text)
    # Split by looking for an instance of 'Supplementary material' (ignoring case)
    # begins a line on its own (followed by any amount of whitespace and then a new line)
    pre_reference = retrieve_text_before_phrase(given_text_cleaned, _reference_regex, 'References')
    pre_supplementary = retrieve_text_before_phrase(pre_reference, _supp_regex, 'Supplementary material')
    pre_conflict = retrieve_text_before_phrase(pre_supplementary, _conf_regex, 'Conflict of interest')
    pre_acknowledgement = retrieve_text_before_phrase(pre_conflict, _ackno_regex, 'Acknowledgments')

    return pre_acknowledgement


def get_info_from_core_paper(paper: dict):
    corpusid = paper['coreId']
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
    doi = paper['doi']
    oai = paper['oai']
    title = clean_title_strings(paper['title'])
    authors = paper['authors']
    url = paper['downloadUrl']
    # provider = return_provider_from_oai(paper['oai'])
    return corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url, oai


def return_provider_from_oai(given_provider: str):
    if given_provider is not None:
        match = re.search(r'oai:(.+):', given_provider)
        if match:
            out = match.group(1)
            return out
        else:
            return None
    else:
        return None


def process_tar_paper_member_lines(paper):
    text = clean_paper_text(paper)

    if text is not None:
        k_word_counts = number_of_keywords(text)
        if any(len(k_word_counts[kword_type]) > 0 for kword_type in k_word_counts):
            corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url, oai = get_info_from_core_paper(
                paper)

            info_df = pd.DataFrame(
                build_output_dict(corpusid, doi, year, k_word_counts, title, authors,
                                  url, _rel_abstract_path, _rel_text_path, language=language, journals=journals,
                                  subjects=subjects, topics=topics, issn=issn, oai=oai))
            return info_df


def process_provider(extracted_provider: str) -> None:
    # TODO: Improve query path handling
    # TODO: +'.tar.xz' This is an artifact to be removed in future versions

    extracted_provider_archive = extracted_provider + '.tar.xz'
    provider_csv = os.path.join(core_paper_info_query_path, extracted_provider_archive + '.csv')

    # Check if already done. Useful for when e.g. cluster fails
    if not os.path.isfile(provider_csv):
        print(f'checking provider {extracted_provider}')
        print(provider_csv)
        provider_df = pd.DataFrame()
        start_time = time.time()
        paper_count = 0
        # find all .json files recursively in nested directories in PATH
        extracted_provider_path = os.path.join(extracted_core_path, extracted_provider)
        for root, dirnames, filenames in os.walk(extracted_provider_path):
            for filename in fnmatch.filter(filenames, '*.json'):
                paper_count += 1

                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as infile:
                    json_data = json.load(infile)
                    paper_df = process_tar_paper_member_lines(json_data)

                    if paper_df is not None:
                        provider_df = pd.concat([paper_df, provider_df])
        provider_df['tar_archive_name'] = extracted_provider_archive
        if provider_df.empty:
            # TODO: This is another artifact
            provider_df['corpusid'] = np.nan
        end_time = time.time()
        provider_df.set_index(['corpusid'], drop=True).to_csv(provider_csv)
        print(
            f'{len(provider_df)} out of {paper_count} papers collected from provider: {extracted_provider_archive}.Took {round((end_time - start_time) / 60, 2)} mins.')

    else:
        print(f'Already checked: {provider_csv}')


def get_relevant_papers_from_download(profile: bool = False):
    provider_list = os.listdir(extracted_core_path)
    print(f'Number of providers to check: {len(provider_list)}')

    # Main archive length: 10251?
    # Each member is a Data provider, see here: https://core.ac.uk/data-providers
    if profile:
        import cProfile
        import pstats
        for provider in provider_list:
            with cProfile.Profile() as pr:
                process_provider(provider)

            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()
    else:
        from multiprocessing import Pool
        pool = Pool()
        pool.map(process_provider, provider_list)


if __name__ == '__main__':
    get_relevant_papers_from_download()
