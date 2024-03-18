import ast
import os
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

from literature_downloads.core import core_download_path
from literature_downloads.core.gather_papers import CQuery, medicine_sort_order, toxic_sort_order


def generic_category_plot(query: CQuery, dict_to_plot: dict, xlabel: str, figsize=(40, 10), sort_var=True, head: int = None):
    import seaborn as sns
    title = f'{xlabel} in {query.name}'
    all_data = pd.DataFrame.from_dict(dict_to_plot, orient='index', columns=['Count'])
    all_data[xlabel] = all_data.index
    if sort_var:
        all_data = all_data.sort_values(by='Count', ascending=False)
        if head is not None:
            all_data = all_data.head(head)
            title = f'Top {head} {title}'

    plt.figure(figsize=figsize)
    sns.barplot(data=all_data, x=xlabel, y='Count')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(query.output_dir, 'plots', title + '.jpg'), dpi=300)
    plt.close()


def get_counts_from_list_column(df, col, capitalize=True):
    ## Count and plot families
    family_counts = defaultdict(int)
    values_to_check = df[col].dropna().values
    for fl in values_to_check:
        fam_list = ast.literal_eval(fl)
        for fam in fam_list:
            family_counts[fam] += 1
    if capitalize:
        for k in list(family_counts.keys()):
            family_counts[k.capitalize()] = family_counts.pop(k)
    return family_counts


def plot_all_info(query: CQuery):
    df = pd.read_csv(query.extracted_paper_csv)
    if not os.path.exists(os.path.join(query.output_dir, 'plots')):
        os.mkdir(os.path.join(query.output_dir, 'plots'))
    ## Count and plot familiesw
    family_counts = get_counts_from_list_column(df, 'plant_family_names_counts')
    generic_category_plot(query, family_counts, 'Plant Families', sort_var=True)
    generic_category_plot(query, family_counts, 'Plant Families', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'fungi_family_names_counts')
    generic_category_plot(query, family_counts, 'Fungi Families', sort_var=True)
    generic_category_plot(query, family_counts, 'Fungi Families', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'lifeform_counts')
    generic_category_plot(query, family_counts, 'Lifeforms', sort_var=True)
    generic_category_plot(query, family_counts, 'Lifeforms', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'plant_genus_names_counts')
    generic_category_plot(query, family_counts, 'Plant Genera', sort_var=True)
    generic_category_plot(query, family_counts, 'Plant Genera', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'fungi_genus_names_counts')
    generic_category_plot(query, family_counts, 'Fungi Genera', sort_var=True)
    generic_category_plot(query, family_counts, 'Fungi Genera', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'plant_species_binomials_counts')
    generic_category_plot(query, family_counts, 'Plant Species', sort_var=True)
    generic_category_plot(query, family_counts, 'Plant Species', figsize=None, sort_var=True, head=30)

    family_counts = get_counts_from_list_column(df, 'fungi_species_binomials_counts')
    generic_category_plot(query, family_counts, 'Fungi Species', sort_var=True)
    generic_category_plot(query, family_counts, 'Fungi Species', figsize=None, sort_var=True, head=30)
    # # Normalise by num species.
    # # This probably needs some family name resolution...
    # normalized_family_counts = dict()
    # for fam in family_counts:
    #     fam_count = len(all_taxa[all_taxa[wcvp_accepted_columns['family']] == fam].index)
    #     normalized_family_counts[fam] = family_counts[fam] / fam_count
    # generic_category_plot(query, normalized_family_counts, 'Normalized Families', sort_var=True)


if __name__ == '__main__':
    # from wcvp_download import get_all_taxa, wcvp_accepted_columns
    # all_taxa = get_all_taxa(accepted=True, ranks=['Species'])
    medicinal_query = CQuery('en_medic_toxic_keywords_final.zip', os.path.join(core_download_path, 'medicinals'),
                             'medicinals.csv',
                             medicine_sort_order,
                             10000)
    medicinal_query.name = 'Medicinal Query'
    plot_all_info(medicinal_query)

    toxic_query = CQuery('en_medic_toxic_keywords_final.zip', os.path.join(core_download_path, 'toxics'),
                         'toxics.csv',
                         toxic_sort_order, 10000)
    toxic_query.name = 'Toxic Query'
    plot_all_info(toxic_query)
