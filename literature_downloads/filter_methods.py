import os
import string
import time
from collections import Counter
from typing import List

import pandas as pd

from literature_downloads import final_en_keyword_dict

query_name = 'en_keywords'


def get_dict_from_res(count_result: Counter, list_to_check: List[str]):
    # start_time = time.time()
    intersection = set(count_result.keys()).intersection(list_to_check)
    out_res = {key: count_result[key] for key in intersection}
    # print("getting number of keywords using intersection: %s seconds ---" % (time.time() - start_time))
    return out_res


def number_of_keywords(given_text: str):
    out_dict = {}
    # start_time = time.time()
    words = [w.strip(string.punctuation).lower() for w in given_text.split()]
    # Species names could be 3 words long due to hybrid characters
    paired_words = [" ".join([words[i], words[i + 1]]) for i in range(len(words) - 1)]
    trio_words = [" ".join([words[i], words[i + 1], words[i + 2]]) for i in range(len(words) - 2)]
    potential_words = words + paired_words + trio_words

    res = Counter(potential_words)
    for k in final_en_keyword_dict:
        out_dict[k] = get_dict_from_res(res, final_en_keyword_dict[k])

    return out_dict


def build_output_dict(corpusid, doi, keyword_counts, title, authors, url, _rel_abstract_path,
                      _rel_text_path, language=None):
    out_dict = {'corpusid': [corpusid], 'DOI': [doi], 'language': language,
                'title': [title], 'authors': [str(authors)], 'oaurl': [url],
                'abstract_path': [os.path.join(_rel_abstract_path, corpusid + '.txt')],
                'text_path': [os.path.join(_rel_text_path, corpusid + '.txt')]}

    for k in keyword_counts:
        out_dict[k + '_counts'] = str(keyword_counts[k])
        out_dict[k + '_total'] = sum(keyword_counts[k].values())
        out_dict[k + '_unique_total'] = len(keyword_counts[k].keys())

    return out_dict


def sort_final_dataframe(df: pd.DataFrame):
    return df.sort_values(
        by=['species_binomials_unique_total', 'family_names_unique_total', 'medicinal_unique_total', 'plants_unique_total',
            'genus_names_unique_total'],
        ascending=False).reset_index(drop=True)
