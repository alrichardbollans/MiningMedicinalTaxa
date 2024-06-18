allowed_links = {'treats_medical_condition': [['Scientific Plant Name', 'Medical Condition'], ['Scientific Fungus Name', 'Medical Condition']],
                 'has_medicinal_effect': [['Scientific Plant Name', 'Medicinal Effect'], ['Scientific Fungus Name', 'Medicinal Effect']]}


def check_is_in_keywords(given_str: str, keyword_dict: dict) -> bool:
    # TODO: Checking in keywords won't work for e.g. names containing authors.
    """
    Check if a given string is in the keywords of a keyword dictionary. (where the dict is an output of get_kword_dict())

    :param given_str: The string to check.
    :param keyword_dict: A dictionary containing the keywords as values.

    :return: True if the given string is found in any of the keywords, False otherwise.
    """
    if given_str != given_str or given_str is None:
        return False
    else:
        for query in keyword_dict:
            if given_str in keyword_dict[query]:
                print(f'{given_str} found in {query}')
                return True
        return False


def check_taxon_is_known_name(given_taxon: str) -> bool:
    # Just check the first word to see if it could be a genus.
    pass


def check_name_and_relation_are_in_text(text: str, taxon_name: str, condition_or_effect: str) -> bool:
    pass


def find_relevant_snippet(text: str, taxon_name: str, condition_or_effect: str) -> str:
    pass


def get_reference(core_id) -> str:
    pass


def parse_model_output_for_app(response: TaxaData, core_id: str):
    pass


def parse_app_output_for_database():
    pass


def parse_app_output_for_finetuning():
    pass
