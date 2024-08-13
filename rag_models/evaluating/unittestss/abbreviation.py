import unittest

from rag_models.evaluating import evaluation_methods


class TestEvaluationMethods(unittest.TestCase):

    def test_abbreviate_single_word(self):
        result = evaluation_methods.abbreviate('Python')
        self.assertEqual(result, 'Python')

    def test_abbreviate_multiple_words(self):
        result = evaluation_methods.abbreviate('Python Charm')
        self.assertEqual(result, 'P. Charm')

        result = evaluation_methods.abbreviate('Python Charm last')
        self.assertEqual(result, 'P. Charm last')

    def test_abbreviate_empty_string(self):
        result = evaluation_methods.abbreviate('')
        self.assertEqual(result, '')

    def test_abbreviate_with_spaces(self):
        with self.assertRaises(ValueError):
            evaluation_methods.abbreviate('  Python  ')

    def test_abbreviate_with_leading_spaces(self):
        with self.assertRaises(ValueError):
            evaluation_methods.abbreviate('  Python Charm')

    def test_abbreviate_with_trailing_spaces(self):
        with self.assertRaises(ValueError):
            evaluation_methods.abbreviate('Python Charm  ')


if __name__ == '__main__':
    unittest.main()
