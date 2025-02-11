import os

import pandas as pd
from wcvpy.wcvp_download import plot_native_number_accepted_taxa_in_regions
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column
_WCVP_VERSION = '12'

def get_mpns_df():
    mpns_input = pd.read_excel(os.path.join('inputs', 'MPNS_v12_names.xlsx'), sheet_name='mpns_v12_acc_names')[['taxon_name']]
    mpns_input.columns = ['given_acc_name']
    acc_name_df = get_accepted_info_from_names_in_column(mpns_input, 'given_acc_name', wcvp_version=_WCVP_VERSION)
    acc_name_df.to_csv(os.path.join('inputs', 'MPNS_v12_acc_sp_names.csv'))
    return acc_name_df[['accepted_species']]

def get_tp_fn_from_annotated_test_data():
    test = pd.read_csv(os.path.join('..', 'evaluating', 'outputs', 'for_testing.csv'))
    fileNames = os.listdir(os.path.join('..', 'evaluating', 'outputs', 'model_errors'))
    true_positives = []
    false_negatives = []
    for f in fileNames:
        if f.endswith("ft_gpt-4o-2024-08-06_personal__Acwijdma_problems.csv"):
            if any(f.startswith(f'{str(t)}_') for t in test['id'].unique().tolist()):
                model_results = pd.read_csv(os.path.join('..', 'evaluating', 'outputs', 'model_errors', f), index_col=0)
                true_positives += model_results['MedCond_tp'].dropna().tolist()
                false_negatives += model_results['MedCond_fn'].dropna().tolist()
    print(f"True positives: {len(true_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    true_positives = list(set(true_positives))
    false_negatives = list(set(false_negatives))
    return true_positives, false_negatives
def main():
    true_positives, false_negatives = get_tp_fn_from_annotated_test_data()


    ## Resolve to species
    def resolve_list_to_clean_df(name_list):
        # Check no duplicate underscores as these are used to seprate names and conditions.
        issues = [print(x) for x in name_list if x.count('_') > 1]
        assert len(issues) == 0

        names_with_medcond = list(set([c.split('_')[0] for c in name_list]))
        name_df = pd.DataFrame(names_with_medcond, columns=['name'])
        acc_name_df = get_accepted_info_from_names_in_column(name_df, 'name', wcvp_version=_WCVP_VERSION)
        acc_name_df = acc_name_df[~acc_name_df['accepted_species'].isna()]
        acc_name_df = acc_name_df.drop_duplicates(subset=['accepted_species'])
        return acc_name_df

    tp_acc_name_df = resolve_list_to_clean_df(true_positives)
    tp_acc_name_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'tp_accepted_species_with_medCond.csv'))
    tp_acc_name_df.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'tp_accepted_species_with_medCond_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(tp_acc_name_df, 'accepted_species', os.path.join('outputs', 'mpns_analysis'),
                                                'tp_accepted_species_with_medCond.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')

    fn_match_acc_name_df = resolve_list_to_clean_df(false_negatives)
    fn_match_acc_name_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'fn_accepted_species_with_medCond.csv'))
    fn_match_acc_name_df.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'fn_accepted_species_with_medCond_summary.csv'))

    all_acc_name_df = resolve_list_to_clean_df(true_positives+false_negatives)
    all_acc_name_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'all_accepted_species_with_medCond.csv'))
    all_acc_name_df.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'all_accepted_species_with_medCond_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(all_acc_name_df, 'accepted_species', os.path.join('outputs', 'mpns_analysis'),
                                                'all_accepted_species_with_medCond.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')


    ## Check against MPNS
    mpns_df = pd.read_csv(os.path.join('inputs', 'MPNS_v12_acc_sp_names.csv'))
    species_not_in_mpns = tp_acc_name_df[~tp_acc_name_df['accepted_species'].isin(mpns_df['accepted_species'].values)]
    species_not_in_mpns.to_csv(os.path.join('outputs', 'mpns_analysis', 'tp_species_not_in_mpns.csv'))
    species_not_in_mpns.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'tp_species_not_in_mpns_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(species_not_in_mpns, 'accepted_species', os.path.join('outputs', 'mpns_analysis'),
                                                'tp_species_not_in_mpns.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')


if __name__ == '__main__':
    # get_mpns_df()
    main()
