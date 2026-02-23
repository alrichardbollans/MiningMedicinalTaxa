import os

import pandas as pd
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from LLM_models.checking_and_summarising_annotations import get_all_human_annotations_for_chunk_id


def main():
    manual_checks = pd.read_csv('manual_outputs.csv')
    correct = manual_checks[manual_checks['decision'] == 'Yes']
    incorrect = manual_checks[manual_checks['decision'] == 'No']
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")

    precision = len(correct) / (len(correct) + len(incorrect))
    print(f"Precision: {precision}")

    for_hparam_tuning = pd.read_csv(os.path.join('..', 'LLM_models', 'evaluating', 'outputs', 'for_hparam_tuning.csv'))
    chunk_ids = for_hparam_tuning['id'].unique().tolist()
    collected_taxa = []
    for chunk_id in chunk_ids:
        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        taxa = human_annotations.taxa
        for t in taxa:
            collected_taxa.append(t.scientific_name)
    collected_taxa = set(collected_taxa)

    names_in_tuning_data_and_deployment = set(correct['taxon_name'].tolist()) & set(collected_taxa)
    print(f"Names in tuning data and deployment: {len(names_in_tuning_data_and_deployment)}")
    acc_correct_data = get_accepted_info_from_names_in_column(correct.rename(columns={'taxon_name':'sci_name'}), 'sci_name', wcvp_version='12')
    acc_correct_data['pairs'] = acc_correct_data['accepted_name'] + '_' + acc_correct_data['medical_condition'].fillna('') + '_' + acc_correct_data['medicinal_effect'].fillna('')
    acc_correct_data['verbatim_pairs'] = acc_correct_data['sci_name'] + '_' + acc_correct_data['medical_condition'].fillna('') + '_' + acc_correct_data['medicinal_effect'].fillna('')
    acc_correct_data.to_csv('acc_correct_data.csv')


if __name__ == '__main__':
    main()
