# TODO: Set up pre evaluation checks to check annotations for things like leading/trailing whitespace
# TODO: Think about how this affects indices of the annotations.
# TODO: Checking in keywords won't work for e.g. names containing authors.

allowed_links = {'treats_medical_condition': [['Scientific Plant Name', 'Medical Condition'], ['Scientific Fungus Name', 'Medical Condition']],
                 'has_medicinal_effect': [['Scientific Plant Name', 'Medicinal Effect'], ['Scientific Fungus Name', 'Medicinal Effect']]}


def leading_trailing_whitespace(given_str: str):
    """
    Remove leading and trailing whitespace from a given string.

    :param given_str: The input string.
    :return: The input string with leading and trailing whitespace removed.
    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            stripped = given_str.strip()
            return stripped
    except AttributeError:
        return given_str


def leading_trailing_punctuation(given_str: str):
    """
    :param given_str: The input string which may contain leading and trailing punctuation.
    :return: The input string with leading and trailing punctuation removed.

    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            stripped = given_str.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return stripped
    except AttributeError:
        return given_str


def lowercase(given_str: str):
    """
    Convert the given string to lowercase.

    :param given_str: The string to be converted to lowercase.
    :return: The lowercase version of the given string.
    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            return given_str.lower()
    except AttributeError:
        return given_str


def check_is_in_keywords(given_str: str, keyword_dict: dict):
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


def clean_strings(given_str: str):
    """
    Clean the given string by removing leading/trailing whitespace,
    leading/trailing punctuation, and converting all characters to lowercase.

    :param given_str: The string to be cleaned.
    :return: The cleaned string.
    """
    return lowercase(leading_trailing_punctuation(leading_trailing_whitespace(given_str)))


def standardise_NER_annotations(annotations: list, keyword_dict: dict):
    """
    Standardizes the NER annotations by cleaning the text strings and checking if they are in the given keyword dictionary.

    :param annotations: List of annotations dictionary.
    :param keyword_dict: Dictionary containing keywords.
    :return: None
    """

    for ann in annotations:
        ann['value']['text'] = clean_strings(ann['value']['text'])
        if ann['value']['label'] in ['Scientific Plant Name', 'Scientific Fungus Name']:
            if not check_is_in_keywords(ann['value']['text'], keyword_dict):
                print(f'Warning: {ann["value"]["label"]} annotation "{ann["value"]["text"]}" is not in keywords.')


def standardise_RE_annotations(annotations: list):
    """
    :param annotations: A list of annotation dictionaries representing links between entities.
    :return: None

    This function takes a list of annotation dictionaries and standardizes the annotations by cleaning the text values of entities involved in the links.
    It also checks the validity of the links and prints a warning message if an invalid link is found.
    """
    for ann in annotations:
        ann['from_entity']['value']['text'] = clean_strings(ann['from_entity']['value']['text'])
        ann['to_entity']['value']['text'] = clean_strings(ann['to_entity']['value']['text'])

        allowed_link = allowed_links[ann['label']]
        for from_entity_label in ann['from_entity']['value']['labels']:
            relevant_links = [c for c in allowed_link if c[0] == from_entity_label]
            if not any(link[1] in ann['to_entity']['value']['labels'] for link in relevant_links):
                print(f'Warning: Invalid link "{ann["label"]}" from "{from_entity_label}" to {ann["to_entity"]["value"]["labels"]}')
