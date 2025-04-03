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
    medcond_true_positives = []
    medcond_false_negatives = []
    ner_true_positives = []
    med_eff_true_positives = []
    for f in fileNames:
        if f.endswith("ft_gpt-4o-2024-08-06_personal__BHfNoQa3_problems.csv"):
            if any(f.startswith(f'{str(t)}_') for t in test['id'].unique().tolist()):
                model_results = pd.read_csv(os.path.join('..', 'evaluating', 'outputs', 'model_errors', f), index_col=0)
                medcond_true_positives += model_results['MedCond_tp'].dropna().tolist()
                medcond_false_negatives += model_results['MedCond_fn'].dropna().tolist()

                ner_true_positives += model_results['NER_tp_in_model'].dropna().tolist()
                med_eff_true_positives += model_results['MedEff_tp'].dropna().tolist()

    # Check no duplicate underscores as these are used to seprate names and conditions.
    issues = [print(x) for x in medcond_true_positives + medcond_false_negatives if x.count('_') > 1]
    assert len(issues) == 0

    tp_names_with_medcond = list(set([c.split('_')[0] for c in medcond_true_positives]))
    fn_names_with_medcond = list(set([c.split('_')[0] for c in medcond_false_negatives]))
    tp_names = list(set([c.split('_')[0] for c in ner_true_positives]))
    tp_med_eff = list(set([c.split('_')[0] for c in med_eff_true_positives]))

    out_df = pd.DataFrame([[len(tp_names), len(tp_names_with_medcond), len(fn_names_with_medcond), len(tp_med_eff)]],
                          columns=['NER TP', 'Med Cond TP', 'Med Cond FN', 'Med Eff TP'])
    out_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'summary.csv'))
    return tp_names_with_medcond, fn_names_with_medcond, tp_med_eff


def main():
    out_folder = os.path.join('outputs', 'mpns_analysis', 'vascular plants')
    true_positives, false_negatives, tp_med_eff = get_tp_fn_from_annotated_test_data()

    ## Resolve to species
    def resolve_list_to_clean_df(name_list):
        name_df = pd.DataFrame(name_list, columns=['name'])
        acc_name_df = get_accepted_info_from_names_in_column(name_df, 'name', wcvp_version=_WCVP_VERSION)
        acc_name_df = acc_name_df[~acc_name_df['accepted_species'].isna()]
        acc_name_df = acc_name_df.drop_duplicates(subset=['accepted_species'])
        return acc_name_df

    tp_acc_name_df = resolve_list_to_clean_df(true_positives)
    tp_acc_name_df.to_csv(os.path.join(out_folder, 'tp_accepted_species_with_medCond.csv'))
    tp_acc_name_df.describe(include='all').to_csv(
        os.path.join(out_folder, 'tp_accepted_species_with_medCond_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(tp_acc_name_df, 'accepted_species', os.path.join(out_folder),
                                                'tp_accepted_species_with_medCond.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')

    tp_med_eff_acc_name_df = resolve_list_to_clean_df(tp_med_eff)
    tp_med_eff_acc_name_df.to_csv(os.path.join(out_folder, 'tp_accepted_species_with_medEff.csv'))
    tp_med_eff_acc_name_df.describe(include='all').to_csv(
        os.path.join(out_folder, 'tp_accepted_species_with_medEff_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(tp_med_eff_acc_name_df, 'accepted_species', os.path.join(out_folder),
                                                'tp_accepted_species_with_medEff.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')

    fn_match_acc_name_df = resolve_list_to_clean_df(false_negatives)
    fn_match_acc_name_df.to_csv(os.path.join(out_folder, 'fn_accepted_species_with_medCond.csv'))
    fn_match_acc_name_df.describe(include='all').to_csv(
        os.path.join(out_folder, 'fn_accepted_species_with_medCond_summary.csv'))

    all_acc_name_df = resolve_list_to_clean_df(true_positives + false_negatives)
    all_acc_name_df.to_csv(os.path.join(out_folder, 'all_accepted_species_with_medCond.csv'))
    all_acc_name_df.describe(include='all').to_csv(
        os.path.join(out_folder, 'all_accepted_species_with_medCond_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(all_acc_name_df, 'accepted_species', os.path.join(out_folder),
                                                'all_accepted_species_with_medCond.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')

    ## Check against MPNS
    mpns_df = pd.read_csv(os.path.join('inputs', 'MPNS_v12_acc_sp_names.csv'))
    species_not_in_mpns = tp_acc_name_df[~tp_acc_name_df['accepted_species'].isin(mpns_df['accepted_species'].values)]
    species_not_in_mpns.to_csv(os.path.join(out_folder, 'tp_species_not_in_mpns.csv'))
    species_not_in_mpns.describe(include='all').to_csv(
        os.path.join(out_folder, 'tp_species_not_in_mpns_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(species_not_in_mpns, 'accepted_species', os.path.join(out_folder),
                                                'tp_species_not_in_mpns.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')


if __name__ == '__main__':
    # get_mpns_df()
    main()
