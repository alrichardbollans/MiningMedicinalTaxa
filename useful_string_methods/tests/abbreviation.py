import unittest

from useful_string_methods import abbreviate_sci_name


class TestEvaluationMethods(unittest.TestCase):

    def test_abbv_hybrids(self):
        h = ["×", "+"]
        for i in h:
            result = abbreviate_sci_name(f'{i} Python')
            self.assertEqual(result, f'{i} Python')

            result = abbreviate_sci_name(f'{i} Python jython')
            self.assertEqual(result, f'{i} P. jython')

    def test_abbreviate_single_word(self):
        result = abbreviate_sci_name('Python')
        self.assertEqual(result, 'Python')

    def test_abbreviate_multiple_words(self):
        result = abbreviate_sci_name('Python Charm')
        self.assertEqual(result, 'P. Charm')

        result = abbreviate_sci_name('Python Charm last')
        self.assertEqual(result, 'P. Charm last')

        result = abbreviate_sci_name('Python   Charm')
        self.assertEqual(result, 'P. Charm')

        result = abbreviate_sci_name('Python  Charm   last')
        self.assertEqual(result, 'P. Charm last')

    def test_abbreviate_empty_string(self):
        result = abbreviate_sci_name('')
        self.assertEqual(result, '')

    def test_abbreviate_with_spaces(self):
        self.assertEqual(abbreviate_sci_name('  Python  '), '  Python  ')
        self.assertEqual(abbreviate_sci_name('  Python Charm'), 'P. Charm')
        self.assertEqual(abbreviate_sci_name('Python Charm  '), 'P. Charm')


if __name__ == '__main__':
    unittest.main()
