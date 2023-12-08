import os
import string
from collections import Counter
from typing import List

import numpy as np

from literature_downloads import get_kword_dict

final_en_keyword_dict = get_kword_dict()


def get_dict_from_res(count_result: Counter, list_to_check: List[str]):
    # start_time = time.time()
    intersection = set(count_result.keys()).intersection(list_to_check)
    out_res = {key: count_result[key] for key in intersection}
    # print("getting number of keywords using intersection: %s seconds ---" % (time.time() - start_time))
    return out_res


def number_of_keywords(given_text: str):
    out_dict = {}

    words = [w.strip(string.punctuation).lower() for w in given_text.split()]
    # Species names could be 3 words long due to hybrid characters
    paired_words = [" ".join([words[i], words[i + 1]]) for i in range(len(words) - 1)]
    trio_words = [" ".join([words[i], words[i + 1], words[i + 2]]) for i in range(len(words) - 2)]
    potential_words = words + paired_words + trio_words

    for k in final_en_keyword_dict:
        out_dict[k] = set(potential_words).intersection(final_en_keyword_dict[k])

    return out_dict


def build_output_dict(corpusid: str, doi: str, year: int, keyword_counts: dict, title: str, authors: List[str], url: str, _rel_abstract_path: str,
                      _rel_text_path: str, language: str = None, journals: List[str] = None, subjects: List[str] = None, topics: List[str] = None,
                      issn: str = None, oai: str = None):
    out_dict = {'corpusid': [corpusid], 'oai': [oai], 'DOI': [doi], 'year': year, 'language': language, 'journals': journals, 'issn': issn,
                'subjects': subjects, 'topics': topics,
                'title': [title], 'authors': [str(authors)], 'oaurl': [url],
                'abstract_path': [os.path.join(_rel_abstract_path, corpusid + '.txt')],
                'text_path': [os.path.join(_rel_text_path, corpusid + '.txt')]}

    for k in keyword_counts:
        unique_counts = len(keyword_counts[k])
        if unique_counts > 0:
            out_dict[k + '_counts'] = str(list(keyword_counts[k]))
        else:
            out_dict[k + '_counts'] = np.nan
        # out_dict[k + '_total'] = sum(keyword_counts[k].values())
        out_dict[k + '_unique_total'] = unique_counts

    return out_dict
