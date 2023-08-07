species_names = ['pubescens']
genus_names = ['Cinchona']
family_names = ['Rubiaceae']
# common_names = []

misc_keywords = ['medicinal', 'medicine', 'herbal', 'remedy', 'drug']  # add other languages

plant_terms = species_names + genus_names + family_names

query_name = '_'.join(misc_keywords + plant_terms)


# search_filtering_terms = ['medicinal', 'plant', 'medicinal plant']

def is_relevant_text(given_text: str) -> bool:
    if given_text is not None:
        # Note order of this can improve optimisation
        if any(x in given_text for x in misc_keywords) and any(x in given_text for x in plant_terms):
            return True
        else:
            return False
    else:
        return False
