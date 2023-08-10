import string

# species_names = ['pubescens']
# genus_names = ['Cinchona']
# family_names = ['Rubiaceae']
# common_names = []

misc_keywords = ['medicinal', 'medicine', 'herbal', 'remedy', 'drug']  # add other languages
misc_paired_keywords = []
# plant_terms = species_names + genus_names + family_names

query_name = '_'.join(misc_keywords + misc_paired_keywords)

def is_relevant_text(given_text: str) -> str:
    # Note order of this can improve optimisation
    words = [w.strip(string.punctuation) for w in given_text.split()]

    first_match = next((string for string in misc_keywords if string in words), None)
    if first_match is None:
        adjacent_words = [" ".join([words[i], words[i + 1]])
                          for i in range(len(words) - 1)]
        first_match = next((string for string in misc_paired_keywords if string in adjacent_words), None)
    return first_match
