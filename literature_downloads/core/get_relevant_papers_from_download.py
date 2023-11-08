import _lzma
import json
import multiprocessing
import os
import re
import sys
import tarfile

import pandas as pd
import time

from literature_downloads import query_name, number_of_keywords, build_output_dict

sys.path.append('../..')

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

zipfile = 'core_2022-03-11_dataset.tar.xz'


def clean_title_strings(given_title: str) -> str:
    ## Also need to fix encodings e.g. \\u27
    fixed_whitespace = ' '.join(given_title.split())
    return fixed_whitespace


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
    title = clean_title_strings(paper['title'])
    authors = paper['authors']
    url = paper['downloadUrl']

    return corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url


def process_tar_member(provider):
    start_time = time.time()
    member_df = pd.DataFrame()
    with tarfile.open(zipfile, 'r') as main_archive:
        provider_file_obj = main_archive.extractfile(provider)
        provider_name = os.path.basename(provider.name)

        with tarfile.open(fileobj=provider_file_obj, mode='r') as sub_archive:
            try:
                members = sub_archive.getmembers()  # Get members will get all files recursively, though deeper archives will need extracting too.
                for i in range(len(members)):
                    m = members[i]
                    if m.name.endswith('.json'):
                        f = sub_archive.extractfile(m)
                        lines = f.readlines()
                        paper = json.loads(lines[0])
                        text = clean_paper_text(paper)

                        if text is not None:
                            k_word_counts = number_of_keywords(text)
                            if any(len(k_word_counts[kword_type].keys()) > 0 for kword_type in k_word_counts):
                                # paper_count += 1
                                corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url = get_info_from_core_paper(paper)

                                info_df = pd.DataFrame(
                                    build_output_dict(corpusid, doi, year, k_word_counts, title, authors,
                                                      url, _rel_abstract_path, _rel_text_path, language=language, journals=journals,
                                                      subjects=subjects, topics=topics, issn=issn))

                                member_df = pd.concat([member_df, info_df])
                    elif m.name.endswith('.xml'):
                        f = sub_archive.extractfile(m)
                        lines = f.readlines()
                    elif '.tar' in m.name:
                        print('Need more recursion')
                        raise ValueError
            except _lzma.LZMAError:
                print(sub_archive)

    member_df['provider_in_tar'] = provider_name
    member_df.set_index(['corpusid'], drop=True).to_csv(os.path.join(core_paper_info_query_path, provider_name + '.csv'))
    end_time = time.time()
    print(f'{len(member_df)} papers collected from provider: {provider_name}. Took {round((end_time - start_time) / 60, 2)} mins')
    return member_df


def get_relevant_papers_from_download():
    print('unzipping main archive')
    # May need to make mode 'r:'
    with tarfile.open(zipfile, 'r') as main_archive:
        # This is slow but useful info. # Main archive length: 10251
        # print(f'Main archive length: {len(main_archive.getnames())}')
        # iterate over members then get all members out of these
        # Each member is a Data provider, see here: https://core.ac.uk/data-providers
        print('unzipped main archive')
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            [pool.apply_async(process_tar_member, args=(member,)) for member in main_archive]


if __name__ == '__main__':
    get_relevant_papers_from_download()
