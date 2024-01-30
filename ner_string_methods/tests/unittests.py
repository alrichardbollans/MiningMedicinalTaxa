import unicodedata
import unittest

from pkg_resources import resource_filename

from literature_downloads import get_kword_dict
from ner_string_methods import retrieve_text_before_phrase, remove_double_spaces_and_break_characters, retrieve_paragraphs_containing_words, \
    remove_HTML_tags, remove_non_ascii_characters, convert_nonascii_to_ascii

_test_output_dir = resource_filename(__name__, 'test_outputs')


class TestGenericMethods(unittest.TestCase):

    def test_retrieving_text_before(self):
        examples = {'This is something that comes\n before\n references \n': 'This is something that comes\n before\n',
                    'This is another example \n where references isnt on its own line': 'This is another example \n where references isnt on its own line',
                    'This is another example where \n 12 REFERENCES \n is on its own line but with line numbers': 'This is another example where \n',
                    ' This is an example with no match...': ' This is an example with no match...', None: None
                    }
        for example in examples:
            self.assertEqual(retrieve_text_before_phrase(example, 'references'), examples[example])
            self.assertEqual(retrieve_text_before_phrase(example, 'References'), examples[example])
            self.assertEqual(retrieve_text_before_phrase(example, 'REFERENCES'), examples[example])

        position_examples = {'This is an example where\n REFERENCES \n matches': 'This is an example where\n', None: None,
                             'This is an example where\n REFERENCES \n matches but it is too early in the text to remove.': 'This is an example where\n REFERENCES \n matches but it is too early in the text to remove.'}
        for example in position_examples:
            self.assertEqual(retrieve_text_before_phrase(example, 'references', check_position_of_phrase=True), position_examples[example])

    def test_remove_double_spaces(self):
        examples = {'This is something that comes\n before\n references \n': 'This is something that comes before references',
                    'This is another example \n   where references isnt on its       own line': 'This is another example where references isnt on its own line',
                    None: None, 'a non space character \xa0': 'a non space character', '\xa0 \xa0 \n': ''
                    }
        for example in examples:
            self.assertEqual(remove_double_spaces_and_break_characters(example), examples[example])

    def test_retrieve_paragraphs_containing_words(self):
        examples = {'This is something with\n no paragraphs references \n': '',
                    'This is another example \n\n   where medicinal is in this paragraph': '   where medicinal is in this paragraph',
                    'This is another example plant_name \n\n   where medicinal is in this paragraph': 'This is another example plant_name \n\n   where medicinal is in this paragraph',
                    None: None, 'medicinal with a non space character \xa0': 'medicinal with a non space character \xa0', '\xa0 \xa0 \n': '',
                    '\n\n': '', ' \n\n ': ''
                    }

        for example in examples:
            self.assertEqual(retrieve_paragraphs_containing_words(example, ['medicinal', 'plant_name']), examples[example])

    def test_example_pipeline(self):
        # Note need to add SCRATCH=Your Repo Path to env variables
        kword_dict = get_kword_dict()
        all_keywords = []
        for dict_value in kword_dict.values():
            all_keywords += list(dict_value)
        example = 'This is an example of some text to be cleaned. This first paragraph will contain a reference to a medicinal plant.\n\n The second paragraph may discuss the Cinchona \n\n This third paragraph \n has no useful information and so is discarded.\n\n Another relevant paragraph discussing cinchona\n\nFinally there will be some references. \n\n References \n 1 Et al.\n\n'
        pre_reference = retrieve_text_before_phrase(example, 'REFERENCES')
        only_useful_paragraphs = retrieve_paragraphs_containing_words(pre_reference, all_keywords)
        final_clean = remove_double_spaces_and_break_characters(only_useful_paragraphs)
        self.assertEqual(final_clean,
                         'This is an example of some text to be cleaned. This first paragraph will contain a reference to a medicinal plant. The second paragraph may discuss the Cinchona Another relevant paragraph discussing cinchona')


class TestRemoveHTMLTags(unittest.TestCase):

    def test_remove_HTML_tags_valid_input(self):
        self.assertEqual(remove_HTML_tags("<h1>Hello, World!</h1>"), "Hello, World!")

    def test_remove_HTML_tags_none_input(self):
        self.assertIsNone(remove_HTML_tags(None), None)

    def test_remove_HTML_tags_multiple_tags(self):
        self.assertEqual(remove_HTML_tags("<h1>Hello,</h1><p>World!</p>"), "Hello,World!")

    def test_remove_HTML_tags_nested_tags(self):
        self.assertEqual(remove_HTML_tags("<div><h1>Hello, World!</h1></div>"), "Hello, World!")

    def test_remove_HTML_tags_empty_string(self):
        self.assertEqual(remove_HTML_tags(""), "")


# create TestNerStringMethodsCleaning class
class TestNerStringMethodsCleaning(unittest.TestCase):

    def test_normalize_text_encoding_valid_input(self):
        """
        Test that it can normalize text encoding of a string
        """
        pairs = [('MÃ¶nk', 'Mnk'), ('dioxeto[3󸀠,4󸀠:3,4]- cyclo-pent[1,2-b]', 'dioxeto[3,4:3,4]- cyclo-pent[1,2-b]'),
                 ('\u0391', ''), ('\udba0\udc20', '')]
        for pair in pairs:
            result = remove_non_ascii_characters(pair[0])
            self.assertIsInstance(result, str)
            self.assertEqual(result, pair[1])

    def test_normalize_text_encoding_empty_string(self):
        """
        Test that it returns empty string as is
        """
        result = remove_non_ascii_characters('')
        self.assertIsInstance(result, str)
        self.assertEqual(result, '')

    def test_normalize_text_encoding_none(self):
        """
        Test that it returns None as None
        """
        result = remove_non_ascii_characters(None)
        self.assertIsNone(result)

    def test_convert_nonascii_to_ascii(self):
        self.assertEqual(convert_nonascii_to_ascii("ñ"), "n")
        self.assertEqual(convert_nonascii_to_ascii("élèvé"), "eleve")
        self.assertEqual(convert_nonascii_to_ascii("café"), "cafe")
        self.assertEqual(convert_nonascii_to_ascii("mañana"), "manana")
        self.assertEqual(convert_nonascii_to_ascii(""), "")
        self.assertEqual(convert_nonascii_to_ascii("normal string with no nonascii"), "normal string with no nonascii")
        self.assertEqual(convert_nonascii_to_ascii('dioxeto[3󸀠,4󸀠:3,4]- cyclo-pent[1,2-b]'), 'dioxeto[3,4:3,4]- cyclo-pent[1,2-b]')
        self.assertEqual(convert_nonascii_to_ascii('\u0391'), 'A')
        self.assertEqual(convert_nonascii_to_ascii('α'), 'a')
        self.assertEqual(convert_nonascii_to_ascii('\udba0\udc20'), '')

if __name__ == '__main__':
    unittest.main()
