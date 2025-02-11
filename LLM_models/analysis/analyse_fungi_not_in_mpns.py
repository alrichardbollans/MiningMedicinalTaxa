import os
import string

from LLM_models.analysis.analyse_instances_not_in_MPNS import get_tp_fn_from_annotated_test_data


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
    # Check no duplicate underscores as these are used to seprate names and conditions.
    issues = [print(x) for x in name_list if x.count('_') > 1]
    assert len(issues) == 0

    names_with_medcond = list(set([c.split('_')[0] for c in name_list]))
    clean_names_with_medcond = [make_clean_binomial(name) for name in names_with_medcond]

    overlaps = set(clean_known_fungi_names).intersection(set(clean_names_with_medcond))
    return overlaps


def your_function():
    true_positives, false_negatives = get_tp_fn_from_annotated_test_data()

    tp_fungi = resolve_list_to_clean_fungi_df(true_positives)

    with open(os.path.join('outputs', 'mpns_analysis', 'fungi','tp_fungi.csv'),
              'w') as f:
        for line in tp_fungi:
            f.write(f"{line}\n")

    fn_fungi = resolve_list_to_clean_fungi_df(false_negatives)

    with open(os.path.join('outputs', 'mpns_analysis', 'fungi','fn_fungi.csv'),
              'w') as f:
        for line in fn_fungi:
            f.write(f"{line}\n")

    all_fungi = resolve_list_to_clean_fungi_df(true_positives + false_negatives)

    with open(os.path.join('outputs', 'mpns_analysis', 'fungi','all_fungi_in_data.csv'),
              'w') as f:
        for line in all_fungi:
            f.write(f"{line}\n")


if __name__ == '__main__':
    with open(os.path.join('..', '..', 'literature_downloads', 'final_keywords_lists', 'fungi_species_binomials_keywords.txt'), 'r') as f:
        fungi_species_binomials = f.read().splitlines()
    ## some minor issues in the species binomials from index fungorum
    issue_lengths = [x for x in fungi_species_binomials if len(x.split(' ')) in [1, 3, 4]]
    assert len(issue_lengths) == 6
    clean_known_fungi_names = [make_clean_binomial(name) for name in fungi_species_binomials]

    your_function()
