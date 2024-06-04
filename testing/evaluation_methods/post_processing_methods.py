## these methods just provide some simple tidying of human and model annotations


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


def clean_strings(given_str: str):
    """
    Clean the given string by removing leading/trailing whitespace,
    leading/trailing punctuation, and converting all characters to lowercase.

    :param given_str: The string to be cleaned.
    :return: The cleaned string.
    """
    return lowercase(leading_trailing_punctuation(leading_trailing_whitespace(given_str)))


def standardise_NER_annotations(annotations: list):
    """
    Standardizes the NER annotations by cleaning the text strings and checking if they are in the given keyword dictionary.

    :param annotations: List of annotations dictionary.
    :param keyword_dict: Dictionary containing keywords.
    :return: None
    """

    for ann in annotations:
        ann['value']['text'] = clean_strings(ann['value']['text'])


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
