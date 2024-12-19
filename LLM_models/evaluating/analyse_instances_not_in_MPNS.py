import os

import pandas as pd
from wcvpy.wcvp_download import plot_native_number_accepted_taxa_in_regions
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column


def main():
    WCVP_VERSION = '12'

    fileNames = os.listdir(os.path.join('outputs', 'model_errors'))
    all_names = []
    for f in fileNames:
        if f.endswith("ft_gpt-4o-2024-08-06_personal__Acwijdma_problems.csv"):
            model_results = pd.read_csv(os.path.join('outputs', 'model_errors', f), index_col=0)
            all_names += model_results['MedCond_tp'].dropna().tolist()
    all_names = list(set(all_names))

    # Check no duplicate underscores as these are used to seprate names and conditions.
    issues = [print(x) for x in all_names if x.count('_') > 1]
    assert len(issues) == 0
    ## Resolve to species
    names_with_medcond = list(set([c.split('_')[0] for c in all_names]))
    name_df = pd.DataFrame(names_with_medcond, columns=['name'])
    acc_name_df = get_accepted_info_from_names_in_column(name_df, 'name', wcvp_version=WCVP_VERSION)
    acc_name_df = acc_name_df[~acc_name_df['accepted_species'].isna()]
    acc_name_df = acc_name_df.drop_duplicates(subset=['accepted_species'])
    acc_name_df.to_csv(os.path.join('outputs', 'mpns_analysis', 'accepted_species_with_medCond.csv'))
    acc_name_df.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'accepted_species_with_medCond_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(acc_name_df, 'accepted_species', os.path.join('outputs', 'mpns_analysis'),
                                                'accepted_species_with_medCond.jpg', wcvp_version=WCVP_VERSION, colormap='inferno')

    ## Check against MPNS
    mpns_df = pd.DataFrame()
    species_not_in_mpns = acc_name_df[~acc_name_df['accepted_species'].isin(mpns_df['accepted_species'].values)]
    species_not_in_mpns.to_csv(os.path.join('outputs', 'mpns_analysis', 'species_not_in_mpns.csv'))
    species_not_in_mpns.describe(include='all').to_csv(os.path.join('outputs', 'mpns_analysis', 'species_not_in_mpns_summary.csv'))

    plot_native_number_accepted_taxa_in_regions(species_not_in_mpns, 'accepted_species', os.path.join('outputs', 'mpns_analysis'),
                                                'species_not_in_mpns.jpg', wcvp_version=WCVP_VERSION, colormap='inferno')

    raise ValueError('Check wcvp version.')


if __name__ == '__main__':
    main()
