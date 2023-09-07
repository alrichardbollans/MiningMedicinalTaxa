import string
from collections import Counter
from typing import List

import pandas as pd
from wcvp_download import get_all_taxa, wcvp_columns

all_taxa = get_all_taxa()
genus_names = all_taxa[wcvp_columns['genus']].unique().tolist()
family_names = all_taxa[wcvp_columns['family']].unique().tolist()
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

query_name = 'en_keywords_genera_families'  # '_'.join(_en_product_keywords)  # + _misc_paired_keywords)

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

_lower_case_plant_names = sorted([x.lower() for x in genus_names + family_names])
print(f'all plant words: {_lower_case_plant_names}')
with open('../plantname_keywords.txt', 'w') as f:
    for line in _lower_case_plant_names:
        f.write(f"{line}\n")

_varied_plantspecific_keywords = sorted(get_varied_forms(plant_specific_keywords))
print(f'all plant key words: {_varied_plantspecific_keywords}')

with open('../plant_keywords.txt', 'w') as f:
    for line in _varied_plantspecific_keywords:
        f.write(f"{line}\n")


def _is_relevant_text(given_text: str) -> str:
    # start_time = time.time()

    # Note order of this can improve optimisation
    words = [w.strip(string.punctuation) for w in given_text.split()]
    lower_words = [x.lower() for x in words]

    first_match = next((string for string in _varied_product_keywords_to_use if string in lower_words), None)
    # if first_match is None:
    #     adjacent_words = [" ".join([words[i], words[i + 1]])
    #                       for i in range(len(words) - 1)]
    #     first_match = next((string for string in _misc_paired_keywords if string in adjacent_words), None)
    # print("getting relevant text takes: %s seconds ---" % (time.time() - start_time))
    return first_match


def number_of_keywords(given_text: str):
    # start_time = time.time()
    words = [w.strip(string.punctuation).lower() for w in given_text.split()]
    res = Counter(words)
    num_product_kwords = {key: res[key] for key in _varied_product_keywords_to_use if key in res}

    num_plantnames = {key: res[key] for key in _lower_case_plant_names if key in res}
    num_plantkwords = {key: res[key] for key in _varied_plantspecific_keywords if key in res}

    # print("getting number of keywords: %s seconds ---" % (time.time() - start_time))
    return num_product_kwords, num_plantnames, num_plantkwords


def sort_final_dataframe(df: pd.DataFrame):
    return df.sort_values(by=['unique_product_keyword_mentions', 'unique_plantname_mentions', 'unique_plantkeyword_mentions'],
                          ascending=False).reset_index(drop=True)
