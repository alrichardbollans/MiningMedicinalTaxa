import unittest

from pkg_resources import resource_filename

from literature_downloads import get_kword_dict
from ner_string_methods import retrieve_text_before_phrase, remove_double_spaces_and_break_characters, retrieve_paragraphs_containing_words

_test_output_dir = resource_filename(__name__, 'test_outputs')


class Test(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
