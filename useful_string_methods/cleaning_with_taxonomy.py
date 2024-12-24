import os
from typing import List

from wcvpy.wcvp_name_matching import get_genus_from_full_name

from useful_string_methods import clean_strings
from wcvpy.wcvp_download import hybrid_characters

scratch_path = os.environ.get('KEWSCRATCHPATH')


def abbreviate_sci_name(name1: str) -> str:
    """
    Return given name with first word abbreviated, if there are multiple words.
    :param name1:
    :return:
    """

    words = name1.split()
    if len(words) < 2:
        return name1
    else:
        if words[0] in hybrid_characters:
            if len(words) < 3:
                return name1
            else:
                words[1] = words[1][0] + '.'
        else:
            words[0] = words[0][0] + '.'
        return ' '.join(words)


def _filter_name_list_using_genus_names(list_of_possible_sci_names: List[str]):
    """
    Filters a given list of possible scientific names, returning a list of scientific names.

    A name is counted as 'scientific' if the first word matches a known genus name.

    :param list_of_possible_sci_names: A list of strings representing possible scientific names.
    :return: A list of strings representing scientific names that match the criteria.
    """

    def _tidy_list(l):
        return set([clean_strings(get_genus_from_full_name(x)) for x in l])

    cleaned_list = _tidy_list(list_of_possible_sci_names)
    tidied_genus_name_file = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                          'tidied_genus_names.txt')
    try:
        with open(tidied_genus_name_file,
                  'r', encoding="utf8") as file:
            _tidied_genus_names = file.read().splitlines()
    except FileNotFoundError as e:
        print(e)

        _genus_names = []
        for g in ['fungi', 'plant']:
            with open(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                   g + '_genus_names_keywords.txt'),
                      'r', encoding="utf8") as file:
                _genus_names.extend(file.read().splitlines())

        _tidied_genus_names = _tidy_list(set(_genus_names))
        with open(tidied_genus_name_file,
                  'w', encoding="utf8") as f:
            for line in _tidied_genus_names:
                f.write(f"{line}\n")

    sci_name_matches = []
    initial_matches = cleaned_list.intersection(_tidied_genus_names)

    sci_name_matches.extend(initial_matches)

    for h in hybrid_characters:
        genus_with_hybrids_iterator = set([h + ' ' + g for g in _tidied_genus_names])
        initial_matches = cleaned_list.intersection(genus_with_hybrids_iterator)

        sci_name_matches.extend(initial_matches)
    sci_name_matches = set(sci_name_matches)

    final_names = []
    for name in list_of_possible_sci_names:
        if clean_strings(get_genus_from_full_name(name)) in sci_name_matches:
            final_names.append(name)
    return final_names


def filter_name_list_with_species_names(list_of_possible_sci_names: List[str]):
    """
    Filters a list of scientific names based on their species names.

    :param list_of_possible_sci_names: A list of strings representing scientific names.
    :return: A list of strings representing scientific names that match species names.
    """

    def tidy_name(x):
        w = clean_strings(x)
        words = w.split()
        if len(words) > 0:
            if words[0] in hybrid_characters:
                return ' '.join(words[:3])
            elif len(words) > 1 and words[1] in hybrid_characters:
                return ' '.join(words[:3])
            else:
                return ' '.join(words[:2])
        else:
            return None
    def _tidy_list(l):
        out = []
        for x in l:
            t_name = tidy_name(x)
            if t_name is not None:
                out.append(t_name)
        return set(out)
    def tidy_and_abbreviate_name(x):
        w = tidy_name(abbreviate_sci_name(x))
        return w

    def _tidy_and_abbreviate_list(l):
        out = []
        for x in l:
            t_name = tidy_and_abbreviate_name(x)
            if t_name is not None:
                out.append(t_name)
        return set(out)

    cleaned_list = set([tidy_name(x) for x in list_of_possible_sci_names])

    abbv_binomial_name_file = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                           'abbreviated_binomial_names.txt')
    try:
        with open(abbv_binomial_name_file,
                  'r', encoding="utf8") as file:
            _abbreviated_binomial_names = file.read().splitlines()
    except FileNotFoundError as e:
        print(e)

        _binomial_names = []
        for g in ['fungi', 'plant']:
            with open(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                   g + '_species_binomials_keywords.txt'),
                      'r', encoding="utf8") as file:
                _binomial_names.extend(file.read().splitlines())

        _abbreviated_binomial_names = _tidy_and_abbreviate_list(set(_binomial_names))

        with open(abbv_binomial_name_file,
                  'w', encoding="utf8") as f:
            for line in _abbreviated_binomial_names:
                f.write(f"{line}\n")

    cleaned_binomial_name_file = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                           'cleaned_binomial_names.txt')
    try:
        with open(cleaned_binomial_name_file,
                  'r', encoding="utf8") as file:
            _cleaned_binomial_names = file.read().splitlines()
    except FileNotFoundError as e:
        print(e)

        _binomial_names = []
        for g in ['fungi', 'plant']:
            with open(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists',
                                   g + '_species_binomials_keywords.txt'),
                      'r', encoding="utf8") as file:
                _binomial_names.extend(file.read().splitlines())

        _cleaned_binomial_names = _tidy_list(set(_binomial_names))

        with open(cleaned_binomial_name_file,
                  'w', encoding="utf8") as f:
            for line in _cleaned_binomial_names:
                f.write(f"{line}\n")


    initial_abbv_matches = cleaned_list.intersection(_abbreviated_binomial_names)
    initial_binomial_matches= cleaned_list.intersection(_cleaned_binomial_names)

    final_names = []
    for name in list_of_possible_sci_names:
        t_name = tidy_name(name)
        if t_name is not None and (t_name in initial_abbv_matches or t_name in initial_binomial_matches):
            final_names.append(name)
    return final_names


def filter_name_list_using_sci_names(list_of_possible_sci_names: List[str]):
    """
    Filters a given list of possible scientific names, returning a list of scientific names.

    A name is counted as 'scientific' if the first word matches a known genus name,
    or if the first word abbreviated + the second word is a binomial name

    :param list_of_possible_sci_names: A list of strings representing possible scientific names.
    :return: A list of strings representing scientific names that match the criteria.
    """

    genus_matches = _filter_name_list_using_genus_names(list_of_possible_sci_names)
    remaining = [c for c in list_of_possible_sci_names if c not in genus_matches]
    if len(remaining) > 0:
        binom_matches = filter_name_list_with_species_names(remaining)
    else:
        binom_matches = []
    return genus_matches + binom_matches
