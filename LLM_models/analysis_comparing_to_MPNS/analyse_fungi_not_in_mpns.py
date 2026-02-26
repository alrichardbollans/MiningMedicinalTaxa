import os
import string

import pandas as pd


def make_clean_binomial(given_string):
    # Clean names for a basic match of binomials
    # Reduce names to two words
    # lowercase words
    # remove punctuation
    out = ' '.join(given_string.split(' ')[:2]).lower()  # first 2 words

    out = out.translate(str.maketrans('', '', string.punctuation))

    return out


## Resolve to species
def resolve_list_to_clean_fungi_df(name_list):

    clean_names_with_medcond = [make_clean_binomial(name) for name in name_list]

    overlaps = set(clean_known_fungi_names).intersection(set(clean_names_with_medcond))
    return overlaps


def your_function():
    correct_outputs = pd.read_csv('../../example_deployment/eval_outputs/correct_outputs.csv')

    tp_fungi = resolve_list_to_clean_fungi_df(correct_outputs['taxon_name'])

    with open(os.path.join('outputs', 'fungi', 'fungi_identified_in_deployment.csv'),
              'w') as f:
        for line in tp_fungi:
            f.write(f"{line}\n")



if __name__ == '__main__':
    with open(os.path.join('..', '..', 'literature_downloads', 'final_keywords_lists', 'fungi_species_binomials_keywords.txt'), 'r') as f:
        fungi_species_binomials = f.read().splitlines()
    ## some minor issues in the species binomials from index fungorum
    issue_lengths = [x for x in fungi_species_binomials if len(x.split(' ')) in [1, 3, 4]]
    assert len(issue_lengths) == 6
    clean_known_fungi_names = [make_clean_binomial(name) for name in fungi_species_binomials]

    your_function()
