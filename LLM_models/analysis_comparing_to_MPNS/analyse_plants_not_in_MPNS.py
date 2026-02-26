import os

import pandas as pd
from wcvpy.wcvp_download import plot_native_number_accepted_taxa_in_regions, get_all_taxa
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
    med_eff_false_negatives = []
    for f in fileNames:
        if f.endswith("ft_gpt-4o-2024-08-06_personal__BHfNoQa3_problems.csv"):
            if any(f.startswith(f'{str(t)}_') for t in test['id'].unique().tolist()):
                model_results = pd.read_csv(os.path.join('..', 'evaluating', 'outputs', 'model_errors', f), index_col=0)
                medcond_true_positives += model_results['MedCond_tp'].dropna().tolist()
                medcond_false_negatives += model_results['MedCond_fn'].dropna().tolist()

                ner_true_positives += model_results['NER_tp_in_model'].dropna().tolist()
                med_eff_true_positives += model_results['MedEff_tp'].dropna().tolist()
                med_eff_false_negatives += model_results['MedEff_fn'].dropna().tolist()

    # Check no duplicate underscores as these are used to seprate names and conditions.
    issues = [print(x) for x in medcond_true_positives + medcond_false_negatives if x.count('_') > 1]
    assert len(issues) == 0

    tp_names_with_medcond = list(set([c.split('_')[0] for c in medcond_true_positives]))
    fn_names_with_medcond = list(set([c.split('_')[0] for c in medcond_false_negatives]))
    tp_names = list(set([c.split('_')[0] for c in ner_true_positives]))
    tp_med_eff = list(set([c.split('_')[0] for c in med_eff_true_positives]))
    fn_med_eff = list(set([c.split('_')[0] for c in med_eff_false_negatives]))

    # out_df = pd.DataFrame([[len(tp_names), len(tp_names_with_medcond), len(fn_names_with_medcond), len(tp_med_eff), len(fn_med_eff)]],
    #                       columns=['NER TP', 'Med Cond TP', 'Med Cond FN', 'Med Eff TP', 'Med Eff FN'])
    # out_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'summary.csv'))
    return tp_names_with_medcond, fn_names_with_medcond, tp_med_eff, fn_med_eff


def compare_annotated_data_with_underlying_pop():
    out_folder = os.path.join('outputs', 'vascular plants', 'annotated_test_data_vs_underlying_pop')

    all_species = get_all_taxa(accepted=True, ranks=['Species'], version=_WCVP_VERSION)
    plot_native_number_accepted_taxa_in_regions(all_species, 'accepted_species', os.path.join(out_folder),
                                                'underlying_distribution.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')
    ## Resolve to species
    def resolve_list_to_clean_df(name_list):
        name_df = pd.DataFrame(name_list, columns=['name'])
        acc_name_df = get_accepted_info_from_names_in_column(name_df, 'name', wcvp_version=_WCVP_VERSION)
        acc_name_df = acc_name_df[~acc_name_df['accepted_species'].isna()]
        acc_name_df = acc_name_df.drop_duplicates(subset=['accepted_species'])
        return acc_name_df

    ### Just do all annotated species
    true_positives, false_negatives, tp_med_eff, fn_med_eff = get_tp_fn_from_annotated_test_data()

    all_annotated_medicinal_taxa_acc_name_df = resolve_list_to_clean_df(true_positives + false_negatives + tp_med_eff + fn_med_eff)
    all_annotated_medicinal_taxa_acc_name_df.to_csv(os.path.join(out_folder, 'all_accepted_species_with_medCond_or_medEff.csv'))
    all_annotated_medicinal_taxa_acc_name_df.describe(include='all').to_csv(
        os.path.join(out_folder, 'all_accepted_species_with_medCond_or_medEff_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(all_annotated_medicinal_taxa_acc_name_df, 'accepted_species', os.path.join(out_folder),
                                                'all_accepted_species_with_medCond_or_medEff.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')


def compare_with_mpns():
    out_folder = os.path.join('outputs', 'vascular plants', 'correct_deployment_data_vs_mpns')

    correct_outputs = pd.read_csv('../../example_deployment/eval_outputs/correct_outputs.csv').rename(columns={'taxon_name': 'sci_name'})

    accepted_correct_outputs = get_accepted_info_from_names_in_column(correct_outputs, 'sci_name', wcvp_version=_WCVP_VERSION)

    accepted_correct_outputs = accepted_correct_outputs.dropna(subset=['accepted_species'])
    accepted_correct_outputs.describe(include='all').to_csv(os.path.join(out_folder, 'correct_outputs_summary.csv'))

    ## Check against MPNS
    mpns_df = pd.read_csv(os.path.join('inputs', 'MPNS_v12_acc_sp_names.csv'))
    species_not_in_mpns = accepted_correct_outputs[~accepted_correct_outputs['accepted_species'].isin(mpns_df['accepted_species'].values)]
    species_not_in_mpns.to_csv(os.path.join(out_folder, 'found_species_not_in_mpns.csv'))
    species_not_in_mpns.describe(include='all').to_csv(
        os.path.join(out_folder, 'found_species_not_in_mpns_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(species_not_in_mpns, 'accepted_species', os.path.join(out_folder),
                                                'found_species_not_in_mpns.jpg', wcvp_version=_WCVP_VERSION, colormap='inferno')

def main():
    # get_mpns_df()
    compare_annotated_data_with_underlying_pop()
    compare_with_mpns()

if __name__ == '__main__':
    main()
