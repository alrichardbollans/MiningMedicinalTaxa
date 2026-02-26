import os

import pandas as pd
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from LLM_models.checking_and_summarising_annotations import get_all_human_annotations_for_chunk_id, repo_path


def get_correct_incorrect_precision(df):
    correct = len(df[df['decision'] == 'Yes'])
    incorrect = len(df[df['decision'] == 'No'])
    precision = correct / (correct + incorrect)
    print(f"Precision: {precision}")
    return [correct, incorrect, correct + incorrect, precision]


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
    dups = manual_checks[manual_checks.duplicated(subset=['json_file', 'taxon_name', 'medical_condition', 'medicinal_effect'], keep=False)]
    if len(dups) > 0:
        raise ValueError(f'Duplicate entries found in manual_outputs.csv: {dups}')
    js_info, corpus_ids = json_info(manual_checks['json_file'].tolist())
    manual_checks['corpusid'] = manual_checks['json_file'].apply(lambda x: corpus_ids[x])
    manual_checks = manual_checks.merge(js_info[['corpusid', 'title']], on='corpusid', how='left')
    manual_checks['verbatim_pairs'] = manual_checks['taxon_name'] + '_' + manual_checks['medical_condition'].fillna('') + '_' + manual_checks[
        'medicinal_effect'].fillna('')

    manual_checks.describe(include='all').to_csv(os.path.join(out_dir, 'manual_outputs_summary.csv'))
    out_df = pd.DataFrame(index=['correct', 'incorrect', 'total', 'precision'])
    out_df['all'] = get_correct_incorrect_precision(manual_checks)
    out_df['med cond'] = get_correct_incorrect_precision(manual_checks[~manual_checks['medical_condition'].isna()])
    out_df['med eff'] = get_correct_incorrect_precision(manual_checks[~manual_checks['medicinal_effect'].isna()])
    out_df.to_csv(os.path.join(out_dir, 'evaluation_results.csv'))
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
    correct.to_csv(os.path.join(out_dir, 'correct_outputs.csv'))
    correct.describe(include='all').to_csv(os.path.join(out_dir, 'correct_outputs_summary.csv'))
    correct_names_in_deployment = set(correct['taxon_name'].tolist())
    names_in_tuning_data_and_deployment = correct_names_in_deployment & set(collected_taxa)
    print(f"Names in tuning data and correct deployment: {len(names_in_tuning_data_and_deployment)}")
    print(f"Correct names deployment: {len(correct_names_in_deployment)}")

    print('cost for the 5 papers: ')

    print(f'Correct medical conditions: {correct['medical_condition'].unique().tolist()}')
    print(f'Correct medicinal effects: {correct['medicinal_effect'].unique().tolist()}')


if __name__ == '__main__':
    main()
