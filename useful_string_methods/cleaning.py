import os
import re

from typing import List


def retrieve_text_before_phrase(given_text: str, simple_string: str, check_position_of_phrase: bool = False) -> str:
    # Split by looking for an instance of simple_string (ignoring case) begins a line on its own (or with line numbers) followed by any amount of
    # whitespace and then a new line
    # Must use re.MULTILINE flag such that the pattern character '^' matches at the beginning of the string and at the beginning of each line
    # (immediately following each newline)
    if given_text:
        my_regex = re.compile(r"^\s*\d*\s*" + simple_string + r"\s*\n", flags=re.IGNORECASE | re.MULTILINE)

        text_split = my_regex.split(given_text, maxsplit=1)

        pre_split = text_split[0]
        if check_position_of_phrase:
            if len(text_split) > 1:
                # At most 1 split occurs, if there has been a split the remainder of the string is returned as the final element of the list.
                # if text after split point is longer than before the split, then revert to given text.
                post_split = text_split[1]
                if len(post_split) > len(pre_split):
                    pre_split = given_text

        return pre_split
    else:
        return given_text


def remove_double_spaces_and_break_characters(given_text: str) -> str:
    '''
    This will simplify a text by removing double spaces and all whitespace characters (e.g. space, tab, newline, return, formfeed).
    See https://stackoverflow.com/a/1546251/8633026
    This may or may not be desirable and should only be used at the end of preprocessing as it removes important characters like \n.
    :param given_text:
    :return:
    '''
    if given_text:
        return " ".join(given_text.split())
    else:
        return given_text


def remove_HTML_tags(given_text: str) -> str:
    """
    Remove HTML tags from the given text.

    :param given_text: The text containing HTML tags.
    :return: The text with HTML tags removed.
    """
    from bs4 import BeautifulSoup

    if given_text is not None:
        # Remove HTML tags using BeautifulSoup
        text = BeautifulSoup(given_text, "html.parser").get_text()
        return text
    else:
        return given_text


def remove_non_ascii_characters(given_text: str) -> str:
    """

    :param given_text: A string that may contain non-ASCII characters.
    :return: A string with any non-ASCII characters removed.

    """
    if given_text is not None:
        return ''.join(i for i in given_text if ord(i) < 128)
    else:
        return given_text


def convert_nonascii_to_ascii(input_str):
    from unidecode import unidecode
    """
    Convert non-ASCII characters to their ASCII equivalents.

    Parameters:
    - input_str (str): The input string containing non-ASCII characters.

    Returns:
    - str: The input string with non-ASCII characters replaced by their ASCII equivalents.
    """
    return unidecode(input_str)


def remove_unneccesary_lines(given_text: str) -> str:
    '''Note this may remove information used to find paragraphs, but can be useful in reducing charcters'''
    without_unneeded_lines = os.linesep.join([s for s in given_text.splitlines() if (not s.isspace())])
    return without_unneeded_lines
