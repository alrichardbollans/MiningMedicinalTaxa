import os
import string
from collections import Counter
from typing import List

import pandas as pd
from wcvp_download import get_all_taxa, wcvp_columns

all_taxa = get_all_taxa()
genus_names = all_taxa[wcvp_columns['genus']].dropna().unique().tolist()
family_names = all_taxa[wcvp_columns['family']].dropna().unique().tolist()
species_binomial_names = all_taxa[all_taxa[wcvp_columns['rank']] == 'Species'][wcvp_columns['name']].dropna().unique().tolist()
_unclean_lifeforms = all_taxa['lifeform_description'].dropna().unique().tolist()
_lifeforms = []
for x in [w.split() for w in _unclean_lifeforms]:
    for y in x:
        w = y.strip(string.punctuation).lower()
        if w not in ['or', 'cl', 'somewhat', 'sometimes']:
            _lifeforms.append(w)
_lifeforms = list(set(_lifeforms))

# Supplement is really caught in a lot of text..

_en_product_keywords = ['medicinal',
                        'medicines',
                        'medicine',
                        'medics',
                        'medic',
                        'pharmacopoeia',
                        'pharmaceuticals',
                        'pharmaceutical',
                        'drug',
                        'drugs',
                        'ethnopharmacology',
                        'remedies',
                        'remedy',
                        'homeopathic',
                        'homeopathy',
                        'homeopath',
                        'homeopaths',
                        'immunoglobulin',
                        'immune',
                        'immunoserum',
                        'oil',
                        'oily',
                        'oils',
                        'candles',
                        'candle',
                        'candlenut',
                        'food',
                        'foods',
                        'foodstuff',
                        'foodstuffs',
                        'food-based',
                        'additive',
                        'additives',
                        'edible',
                        'inedible',
                        'beverages',
                        'beverage',
                        'drink',
                        'drinks',
                        'drinking',
                        'tea',
                        'alcohol',
                        'alcoholic',
                        'hydroalcoholic',
                        'cosmetics',
                        'cosmetic',
                        'shampoo',
                        'shampoos',
                        'shampooing',
                        'toxic',
                        'toxicity',
                        'toxicological',
                        'endotoxin',
                        'toxins',
                        'toxin',
                        'endotoxins',
                        'supplement',
                        'supplements',
                        'supplementary',
                        'antioxidants',
                        'antioxidant',
                        'nutraceuticals',
                        'nutraceutical']

query_name = 'en_keywords_spbinomials_genera_families'  # '_'.join(_en_product_keywords)  # + _misc_paired_keywords)

plant_specific_keywords = ['herbals',
                           'herbal', 'botany', 'ethnobotany',
                           ] + _lifeforms


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
    return forms


def get_varied_forms(list_of_words) -> List:
    forms = []
    for w in list_of_words:
        forms += get_varied_form_of_word(w)
    out = [x.lower() for x in forms]
    return list(set(out))


_varied_product_keywords = get_varied_forms(_en_product_keywords)
words_to_exclude = ['add', 'drunkard']
_varied_product_keywords_to_use = sorted([x for x in _varied_product_keywords if x not in words_to_exclude])
print(f'all variations of keywords: {_varied_product_keywords_to_use}')

with open('../product_keywords.txt', 'w') as f:
    for line in _varied_product_keywords_to_use:
        f.write(f"{line}\n")

_lower_case_genus_names = sorted([x.lower() for x in genus_names])
print(f'all genus words: {_lower_case_genus_names}')
with open('../genusname_keywords.txt', 'w') as f:
    for line in _lower_case_genus_names:
        f.write(f"{line}\n")

_lower_case_family_names = sorted([x.lower() for x in family_names])
print(f'all family words: {_lower_case_family_names}')
with open('../familyname_keywords.txt', 'w') as f:
    for line in _lower_case_family_names:
        f.write(f"{line}\n")

_lower_case_binom_names = sorted([x.lower() for x in species_binomial_names])
# print(f'all sp binom words: {_lower_case_binom_names}')
with open('../sp_binomname_keywords.txt', 'w') as f:
    for line in _lower_case_binom_names:
        f.write(f"{line}\n")

_varied_plantspecific_keywords = sorted(get_varied_forms(plant_specific_keywords))
print(f'all plant key words: {_varied_plantspecific_keywords}')

with open('../plant_keywords.txt', 'w') as f:
    for line in _varied_plantspecific_keywords:
        f.write(f"{line}\n")


def number_of_keywords(given_text: str):
    # start_time = time.time()
    words = [w.strip(string.punctuation).lower() for w in given_text.split()]
    res = Counter(words)
    num_product_kwords = {key: res[key] for key in _varied_product_keywords_to_use if key in res}
    num_familynames = {key: res[key] for key in _lower_case_family_names if key in res}
    num_genusnames = {key: res[key] for key in _lower_case_genus_names if key in res}
    num_plantkwords = {key: res[key] for key in _varied_plantspecific_keywords if key in res}
    # Species names could be 3 words long due to hybrid characters
    paired_words = [" ".join([words[i], words[i + 1]]) for i in range(len(words) - 1)]
    trio_words = [" ".join([words[i], words[i + 1], words[i + 2]]) for i in range(len(words) - 2)]
    potential_binomials = paired_words + trio_words
    paired_res = Counter(potential_binomials)
    num_sp_binomials = {key: paired_res[key] for key in _lower_case_binom_names if key in paired_res}
    # print("getting number of keywords: %s seconds ---" % (time.time() - start_time))
    return num_product_kwords, num_genusnames, num_familynames, num_sp_binomials, num_plantkwords


def build_output_dict(corpusid, doi, total_product_kword_mentions, num_unique_product_kwords, product_kwords_dict,
                      total_genusname_mentions, num_unique_genusnames, genusnames_dict, total_familyname_mentions, num_unique_familynames,
                      familynames_dict,
                      total_species_mentions, unique_species_mentions, species_dict,
                      total_plantkeyword_mentions,
                      num_unique_plantkeywords, plantkwords_dict, title, authors, url, _rel_abstract_path, _rel_text_path, language=None):
    return {'corpusid': [corpusid], 'DOI': [doi], 'language': language,
            'total_product_keyword_mentions': total_product_kword_mentions,
            'unique_product_keyword_mentions': num_unique_product_kwords,
            'product_keyword_count': str(product_kwords_dict),
            'total_genus_mentions': total_genusname_mentions,
            'unique_genus_mentions': num_unique_genusnames,
            'genus_counts': str(genusnames_dict),
            'total_family_mentions': total_familyname_mentions,
            'unique_family_mentions': num_unique_familynames,
            'family_counts': str(familynames_dict),
            'total_species_mentions': total_species_mentions,
            'unique_species_mentions': unique_species_mentions,
            'species_counts': str(species_dict),
            'total_plant_keyword_mentions': total_plantkeyword_mentions,
            'unique_plant_keyword_mentions': num_unique_plantkeywords,
            'plant_keyword_count': str(plantkwords_dict),
            'title': [title], 'authors': [str(authors)], 'oaurl': [url],
            'abstract_path': [os.path.join(_rel_abstract_path, corpusid + '.txt')],
            'text_path': [os.path.join(_rel_text_path, corpusid + '.txt')]}


def sort_final_dataframe(df: pd.DataFrame):
    return df.sort_values(
        by=['unique_species_mentions', 'unique_family_mentions', 'unique_product_keyword_mentions', 'unique_plant_keyword_mentions',
            'unique_genus_mentions'],
        ascending=False).reset_index(drop=True)