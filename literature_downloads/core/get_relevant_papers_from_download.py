import _lzma
import json
import multiprocessing
import os
import re
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import time

from tqdm import tqdm

sys.path.append('../..')
from literature_downloads import query_name, number_of_keywords, build_output_dict

scratch_path = os.environ.get('SCRATCH')

core_project_path = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'core')
core_download_path = os.path.join(core_project_path, 'downloads')
core_abstracts_path = os.path.join(core_download_path, 'abstracts')
_rel_abstract_path = os.path.relpath(core_abstracts_path, scratch_path)

core_text_path = os.path.join(core_download_path, 'text')
_rel_text_path = os.path.relpath(core_text_path, scratch_path)

core_paper_info_path = os.path.join(core_download_path, 'paper_info')
core_paper_info_query_path = os.path.join(core_paper_info_path, query_name)
for p in [core_download_path, core_text_path, core_paper_info_path, core_abstracts_path, core_paper_info_query_path]:
    if not os.path.exists(p):
        os.mkdir(p)

CORE_TAR_FILE = 'core_2022-03-11_dataset.tar.xz'


def clean_title_strings(given_title: str) -> str:
    ## Also need to fix encodings e.g. \\u27
    if given_title is not None:
        fixed_whitespace = ' '.join(given_title.split())
        return fixed_whitespace
    else:
        return given_title


def retrieve_text_before_simple_phrase(given_text: str, simple_string: str) -> str:
    my_regex = r"^" + re.escape(simple_string) + r"\s*\n"
    # Split by looking for an instance of given_text (ignoring case) begins a line on its own (followed by any amount of whitespace and then a new line)
    # Must use re.MULTILINE flag such that the pattern character '^' matches at the beginning of the string and at the beginning of each line (immediately following each newline)
    text_split = re.split(my_regex, given_text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)
    pre_split = text_split[0]
    if len(text_split) > 1:
        post_split = text_split[
            1]  # If maxsplit is nonzero, at most maxsplit splits occur, and the remainder of the string is returned as the final element of the list
        # if text after split point is longer than before, then revert.
        if len(post_split) > len(pre_split):
            pre_split = given_text

    return pre_split


def clean_paper_text(paper: dict) -> str:
    text = paper['fullText']
    if text is None:
        return None

    # Split by looking for an instance of 'Supplementary material' (ignoring case)
    # begins a line on its own (followed by any amount of whitespace and then a new line)
    pre_reference = retrieve_text_before_simple_phrase(text, 'References')
    pre_supplementary = retrieve_text_before_simple_phrase(pre_reference, 'Supplementary material')
    pre_conflict = retrieve_text_before_simple_phrase(pre_supplementary, 'Conflict of interest')
    pre_acknowledgement = retrieve_text_before_simple_phrase(pre_conflict, 'Acknowledgments')

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


def process_tar_member(provider):
    start_time = time.time()
    provider_df = pd.DataFrame()
    total_paper_count = 0
    with tarfile.open(CORE_TAR_FILE, 'r') as main_archive:
        provider_file_obj = main_archive.extractfile(provider)
        tar_archive_name = os.path.basename(provider.name)
        provider_csv = os.path.join(core_paper_info_query_path, tar_archive_name + '.csv')
        # Check if already done. Useful for when e.g. cluster fails
        if not os.path.isfile(provider_csv):
            with tarfile.open(fileobj=provider_file_obj, mode='r') as sub_archive:
                try:
                    members = sub_archive.getmembers()  # Get members will get all files recursively, though deeper archives will need extracting too.
                    for i in range(len(members)):
                        m = members[i]
                        if m.name.endswith('.json'):
                            total_paper_count += 1
                            f = sub_archive.extractfile(m)
                            lines = f.readlines()
                            paper = json.loads(lines[0])
                            if len(lines) > 1:
                                raise ValueError('Unexpected number of lines in archive')
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

                                    provider_df = pd.concat([provider_df, info_df])
                        elif m.name.endswith('.xml'):
                            f = sub_archive.extractfile(m)
                            lines = f.readlines()
                        elif '.tar' in m.name:
                            print('Need more recursion')
                            raise ValueError
                except _lzma.LZMAError:
                    print(f'LZMAError for: {sub_archive}')

        else:
            print(f'Already checked: {provider_csv}')
    if len(provider_df.index) > 0:
        provider_df['tar_archive_name'] = tar_archive_name
        provider_df.set_index(['corpusid'], drop=True).to_csv(provider_csv)
    end_time = time.time()
    print(
        f'{len(provider_df)} out of {total_paper_count} papers collected from provider: {tar_archive_name}. Took {round((end_time - start_time) / 60, 2)} mins.')
    return provider_df


def get_relevant_papers_from_download():
    print('unzipping main archive')
    with tarfile.open(CORE_TAR_FILE, 'r') as main_archive:
        # This is slow but useful info. # Main archive length: 10251
        # print(f'Main archive length: {len(main_archive.getnames())}')
        # names = main_archive.getnames()
        # iterate over members then get all members out of these
        # Each member is a Data provider, see here: https://core.ac.uk/data-providers
        print('unzipped main archive')
        tasks = []
        with multiprocessing.Pool(1) as pool:
            for member in main_archive:

                tasks.append(pool.apply_async(process_tar_member, args=(member,)))

            # Remaining partial batch
            for task in tasks:
                task.get()


if __name__ == '__main__':
    get_relevant_papers_from_download()
