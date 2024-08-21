import os
from typing import List

from wcvpy.wcvp_name_matching import get_genus_from_full_name

from useful_string_methods import clean_strings

scratch_path = os.environ.get('KEWSCRATCHPATH')

def filter_name_list_using_sci_names(list_of_possible_sci_names: List[str]):
    """
    Filters a given list of possible scientific names, returning a new list of scientific names.

    A name is counted as 'scientific' if the first word matches a known genus name.

    :param list_of_possible_sci_names: A list of strings representing possible scientific names.
    :return: A list of strings representing scientific names that match the criteria.
    """
    from wcvpy.wcvp_download import hybrid_characters

    def _tidy_list(l):
        return set([clean_strings(get_genus_from_full_name(x)) for x in l])
    cleaned_list = _tidy_list(list_of_possible_sci_names)

    sci_name_matches = []
    _genus_names = []
    for g in ['fungi', 'plant']:
        with open(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists', g + '_genus_names_keywords.txt'),
                  'r') as file:
            _genus_names.extend(file.read().splitlines())

    _genus_names = _tidy_list(_genus_names)
    initial_matches = cleaned_list.intersection(_genus_names)

    sci_name_matches.extend(initial_matches)

    for h in hybrid_characters:
        genus_with_hybrids_iterator = set([h + ' ' + g for g in _genus_names])
        initial_matches = cleaned_list.intersection(genus_with_hybrids_iterator)

        sci_name_matches.extend(initial_matches)
    sci_name_matches = set(sci_name_matches)

    final_names = []
    for name in list_of_possible_sci_names:
        if clean_strings(get_genus_from_full_name(name)) in sci_name_matches:
            final_names.append(name)
    return final_names