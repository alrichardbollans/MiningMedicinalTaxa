import os
import string

import pandas as pd
from typing import List

query_name = 'en_medic_toxic_keywords'
scratch_path = os.environ.get('KEWSCRATCHPATH')

_keyword_file_tag = '_keywords.txt'
def get_kword_dict():
    """
    Get the keyword dictionary.

    This method reads a list of keywords from text files and stores them in a dictionary.
    Each text file corresponds to a specific query aspect and contains a list of keywords.

    :return: A dictionary where the keys are query names and the values are sets of keywords.
    """
    # TODO: Could store this in a folder depending on query name
    # Read keywords into dict once they have been produced
    word_dict = {}

    # Get list of txt files in folder
    folder = os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists')
    txt_files = [f for f in os.listdir(folder) if f.endswith(_keyword_file_tag)]

    for f in txt_files:
        # Open each text file
        with open(os.path.join(folder, f), encoding='utf-8') as file:
            # Read content into list
            words = file.read().splitlines()

        # Add to dictionary with filename as key
        word_dict[f.replace(_keyword_file_tag, '')] = set(words)

    return word_dict


def summarise_keywords(keyword_dict: dict, name: str):
    out_dict = {}
    for query in keyword_dict:
        out_dict[query] = [len(keyword_dict[query])]
    out_df = pd.DataFrame.from_dict(out_dict, orient='index')
    out_df.loc['total'] = out_df.sum()
    out_df.to_csv(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists', 'summaries',
                               name + '_keywords_summary.csv'))

def _generate_keywords():
    # to install see: https://github.com/alrichardbollans/wcvpy
    from wcvpy.wcvp_download import get_all_taxa, wcvp_columns

    words_to_exclude = [_x.lower().strip() for _x in
                        pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'list_keywords.xlsx'),
                                      sheet_name='Excluded')['Excluded keywords'].tolist()]

    ### Taxa specific
    # IPNI
    _ipni_df = pd.read_csv(
        os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'ipni_flat.csv'))

    _ipni_df = _ipni_df[_ipni_df['citation_type'] != 'miscauto']

    _ipni_families = _ipni_df['family'].dropna().unique().tolist()
    _ipni_genera = _ipni_df['genus'].dropna().unique().tolist()
    _ipni_binomials = _ipni_df[_ipni_df['rank'] == 'spec.']['full_name_without_family_and_authors'].dropna().unique().tolist()

    # PNAPS
    _pnaps_df = pd.read_csv(
        os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'PNAPs.csv'))
    _pnaps_df['simplified_names'] = _pnaps_df['non_sci_name'].apply(lambda x: ' '.join(x.split()[:2]))
    _pnaps = _pnaps_df['simplified_names'].unique().tolist()

    # Plant Checklist
    _all_taxa = get_all_taxa(version='12')
    _genus_names = _all_taxa[wcvp_columns['genus']].dropna().unique().tolist() + _ipni_genera
    _family_names = _all_taxa[wcvp_columns['family']].dropna().unique().tolist() + _ipni_families
    _species_binomial_names = _all_taxa[_all_taxa[wcvp_columns['rank']] == 'Species'][
        wcvp_columns['name']].dropna().unique().tolist()
    _species_binomial_names = _species_binomial_names + _pnaps + _ipni_binomials

    # Fungi
    _fungi_species_df = pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'FungusNames.xlsx'),
                                      sheet_name='SpeciesNames')
    _fungi_species_binomial_names = _fungi_species_df['NAME OF FUNGUS'].dropna().unique().tolist()

    _fungi_genus_df = pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'FungusNames.xlsx'),
                                    sheet_name='GenusNames')
    _fungi_genus_names = _fungi_genus_df['NAME OF FUNGUS'].dropna().unique().tolist()

    _fungi_family_df = pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'FungusNames.xlsx'),
                                     sheet_name='FamilyNames')
    _fungi_family_names = _fungi_family_df['NAME OF FUNGUS'].dropna().unique().tolist()

    ### Lifeforms
    _unclean_lifeforms = _all_taxa['lifeform_description'].dropna().unique().tolist()
    _lifeforms = []
    for _x in [w.split() for w in _unclean_lifeforms]:
        for _y in _x:
            _w = _y.strip(string.punctuation).lower()
            if _w not in ['or', 'cl', 'somewhat', 'sometimes']:
                _lifeforms.append(_w)
    _lifeforms = list(set(_lifeforms))

    def _tidy_list(l, _words_to_exclude=None) -> List:
        """
        Tidy a list by converting all elements to lowercase, removing leading and trailing whitespace,
        and removing any elements that match the provided exclusion list.

        :param l: The list to tidy.
        :param _words_to_exclude: The list of words to exclude from the tidied list. Defaults to an empty list if not provided.
        :return: The tidied list, with elements converted to lowercase, whitespace removed, and excluded words removed.

        Example:
            >>> _tidy_list(["   apple  ", "ORANGE", " grape ", " Banana "], _words_to_exclude=["apple", "banana"])
            ["orange", "grape"]

        """
        if _words_to_exclude is None:
            _words_to_exclude = []
        return list(sorted(set([x.lower().strip() for x in l if x.lower() not in _words_to_exclude])))

    import inflect

    inflect_p = inflect.engine()

    ### Products and plants
    def get_varied_form_of_word(given_word: str) -> List:
        # use nltk to add word variations
        from nltk.corpus import wordnet as wn

        forms = []  # We'll store the derivational forms in a set to eliminate duplicates
        for happy_lemma in wn.lemmas(given_word):  # for each "happy" lemma in WordNet
            forms.append(happy_lemma.name())  # add the lemma itself
            for related_lemma in happy_lemma.derivationally_related_forms():  # for each related lemma
                forms.append(related_lemma.name())  # add the related lemma
        if given_word not in forms:
            forms.append(given_word)

        # Now get plural and singulars
        for f in forms[:]:
            # Fix fungi pluralisation
            if f.endswith('fungi'):
                forms.append(f.replace('fungi', 'fungus'))
            elif f.endswith('fungus'):
                forms.append(f.replace('fungus', 'fungi'))
            else:
                pl = inflect_p.plural(f)
                if pl:
                    forms.append(pl)
                sing = inflect_p.singular_noun(f)
                if sing:
                    forms.append(sing)

        return forms

    def get_varied_forms(list_of_words) -> List:
        forms = []
        for w in list_of_words:
            forms += get_varied_form_of_word(w)
        out = [x.lower() for x in forms]
        return _tidy_list(out, _words_to_exclude=words_to_exclude)

    def _get_keywords_from_df(df: pd.DataFrame):
        d = {}
        df['Keyword Type'].ffill(inplace=True)

        for k in df['Keyword Type']:
            d[k.lower().strip()] = get_varied_forms(df[df['Keyword Type'] == k]['English words'].dropna().apply(
                lambda xl: xl.lower().strip()).unique().tolist())

        return d

    _product_keywords_df = pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'list_keywords.xlsx'),
                                         sheet_name='Product related')
    _product_keyword_dict = _get_keywords_from_df(_product_keywords_df)

    _dual_product_keywords_df = pd.read_excel(
        os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'list_keywords.xlsx'),
        sheet_name='Dual Keywords')
    dual_product_keywords_dict = _get_keywords_from_df(_dual_product_keywords_df)

    _kingdom_kwords_df = pd.read_excel(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'inputs', 'list_keywords.xlsx'),
                                       sheet_name='Kingdom specific')
    _kingdom_specific_keyword_dict = _get_keywords_from_df(_kingdom_kwords_df)
    _kingdom_specific_keyword_dict['lifeform'] = get_varied_forms(_lifeforms)

    out_keyword_dict = {'plant_family_names': _tidy_list(_family_names), 'plant_genus_names': _tidy_list(_genus_names),
                        'plant_species_binomials': _tidy_list(_species_binomial_names),
                        'fungi_family_names': _tidy_list(_fungi_family_names), 'fungi_genus_names': _tidy_list(_fungi_genus_names),
                        'fungi_species_binomials': _tidy_list(_fungi_species_binomial_names)
                        }
    out_keyword_dict.update(_product_keyword_dict)
    out_keyword_dict.update(_kingdom_specific_keyword_dict)
    out_keyword_dict.update(dual_product_keywords_dict)

    for _fk in out_keyword_dict:
        with open(os.path.join(scratch_path, 'MedicinalPlantMining', 'literature_downloads', 'final_keywords_lists', _fk + _keyword_file_tag),
                  'w') as f:
            for line in out_keyword_dict[_fk]:
                f.write(f"{line}\n")
if __name__ == '__main__':
    _generate_keywords()

    summarise_keywords(get_kword_dict(), 'en_medic_toxic_keywords')
