import string

from typing import List
from wcvp_download import get_all_taxa, wcvp_columns

all_taxa = get_all_taxa()
genus_names = all_taxa[wcvp_columns['genus']].unique().tolist()
family_names = all_taxa[wcvp_columns['family']].unique().tolist()
# common_names = []
# species_names = ['pubescens']


_en_keywords = ['medicinal',
                'medicines',
                'medicine',
                'medics',
                'medic',
                'pharmacopoeia',
                'pharmaceuticals',
                'pharmaceutical',
                'ethnobotany',
                'ethnopharmacology',
                'drug',
                'drugs',
                'herbals',
                'herbal',
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


query_name = 'en_keywords_genera_families'#'_'.join(_en_keywords)  # + _misc_paired_keywords)


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


def get_varied_forms() -> set:
    forms = []
    for w in _en_keywords:
        forms += get_varied_form_of_word(w)

    return set(forms)


_varied_keywords = list(get_varied_forms())
words_to_exclude = ['add']
_varied_keywords_to_use = [x for x in _varied_keywords if x not in words_to_exclude]
_all_keywords = _varied_keywords_to_use + genus_names + family_names
print(f'all variations of keywords: {_varied_keywords}')



def is_relevant_text(given_text: str) -> str:
    # Note order of this can improve optimisation
    words = [w.strip(string.punctuation) for w in given_text.split()]

    first_match = next((string for string in _all_keywords if string in words), None)
    # if first_match is None:
    #     adjacent_words = [" ".join([words[i], words[i + 1]])
    #                       for i in range(len(words) - 1)]
    #     first_match = next((string for string in _misc_paired_keywords if string in adjacent_words), None)
    return first_match
