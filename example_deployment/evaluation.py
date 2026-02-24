import os

import pandas as pd
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from LLM_models.checking_and_summarising_annotations import get_all_human_annotations_for_chunk_id, repo_path


def get_correct_incorrect_precision(df):
    correct = df[df['decision'] == 'Yes']
    incorrect = df[df['decision'] == 'No']
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")

    precision = len(correct) / (len(correct) + len(incorrect))
    print(f"Precision: {precision}")

def json_info(json_filenames):
    top_15_hits = pd.read_csv(
        f'{repo_path}/MedicinalPlantMining/literature_downloads/core/downloads/medicinals_top_10000/medicinals_top_10000.csv', index_col=0).head(15)

    corpus_ids = {}
    for j in json_filenames:

        corpus_ids[j] = int(j.split('.json')[0])
    five_hits = top_15_hits[top_15_hits['corpusid'].isin(corpus_ids.values())]

    return five_hits, corpus_ids

def main():
    out_dir = 'eval_outputs'
    manual_checks = pd.read_csv('manual_outputs.csv')
    dups = manual_checks[manual_checks.duplicated(subset=['json_file','taxon_name','medical_condition','medicinal_effect'], keep=False)]
    if len(dups) > 0:
        raise ValueError(f'Duplicate entries found in manual_outputs.csv: {dups}')
    js_info, corpus_ids = json_info(manual_checks['json_file'].tolist())
    manual_checks['corpusid'] = manual_checks['json_file'].apply(lambda x: corpus_ids[x])
    manual_checks = manual_checks.merge(js_info[['corpusid', 'title']], on='corpusid', how='left')
    manual_checks['verbatim_pairs'] = manual_checks['taxon_name'] + '_' + manual_checks['medical_condition'].fillna('') + '_' + manual_checks[
        'medicinal_effect'].fillna('')

    manual_checks.describe(include='all').to_csv(os.path.join(out_dir,'manual_outputs_summary.csv'))
    print('all')
    get_correct_incorrect_precision(manual_checks)
    print('med cond')
    get_correct_incorrect_precision(manual_checks[~manual_checks['medical_condition'].isna()])
    print('med eff')
    get_correct_incorrect_precision(manual_checks[~manual_checks['medicinal_effect'].isna()])

    for_hparam_tuning = pd.read_csv(os.path.join('..', 'LLM_models', 'evaluating', 'outputs', 'for_hparam_tuning.csv'))
    chunk_ids = for_hparam_tuning['id'].unique().tolist()
    collected_taxa = []
    for chunk_id in chunk_ids:
        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        taxa = human_annotations.taxa
        for t in taxa:
            collected_taxa.append(t.scientific_name)
    collected_taxa = set(collected_taxa)

    correct = manual_checks[manual_checks['decision'] == 'Yes']
    correct_names_in_deployment = set(correct['taxon_name'].tolist())
    names_in_tuning_data_and_deployment = correct_names_in_deployment & set(collected_taxa)
    print(f"Names in tuning data and correct deployment: {len(names_in_tuning_data_and_deployment)}")
    print(f"Correct names deployment: {len(correct_names_in_deployment)}")
    acc_correct_data = get_accepted_info_from_names_in_column(correct.rename(columns={'taxon_name': 'sci_name'}), 'sci_name', wcvp_version='12')
    acc_correct_data['pairs'] = acc_correct_data['accepted_name'] + '_' + acc_correct_data['medical_condition'].fillna('') + '_' + acc_correct_data[
        'medicinal_effect'].fillna('')
    acc_correct_data.to_csv(os.path.join(out_dir,'acc_correct_data.csv'))
    acc_correct_data.describe(include='all').to_csv(os.path.join(out_dir,'acc_correct_data_summary.csv'))

    print('cost for the 5 papers: $2.86')


if __name__ == '__main__':
    main()
