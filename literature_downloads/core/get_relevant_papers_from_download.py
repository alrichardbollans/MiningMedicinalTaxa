import json
import os
import re
import sys

import pandas as pd
from tqdm import tqdm
from typing import List, Union

sys.path.append('../..')

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


def get_relevant_papers_from_download():
    paper_df = pd.DataFrame()

    import tarfile
    from literature_downloads import query_name, number_of_keywords, build_output_dict

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
                        text = clean_paper_text(paper)

                        if text is not None:
                            k_word_counts = number_of_keywords(text)
                            if any(len(k_word_counts[kword_type].keys()) > 0 for kword_type in k_word_counts):
                                paper_count += 1
                                corpusid, language, journals, subjects, topics, year, issn, doi, title, authors, url = get_info_from_core_paper(paper)

                                info_df = pd.DataFrame(
                                    build_output_dict(corpusid, doi, year, k_word_counts, title, authors,
                                                      url, _rel_abstract_path, _rel_text_path, language=language, journals=journals,
                                                      subjects=subjects, topics=topics, issn=issn))

                                paper_df = pd.concat([paper_df, info_df])

                paper_df.to_csv(os.path.join(core_paper_info_path, query_name + '.csv'))
    return paper_df


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


if __name__ == '__main__':
    # # load_texts_from_id(81695610)
    # save_texts_from_ids([81695610, 41338455])
    get_relevant_papers_from_download()
