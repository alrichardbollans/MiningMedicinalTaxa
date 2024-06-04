import unittest
from useful_string_methods import simple_pdf_parsing


class TestSimplePdfParsing(unittest.TestCase):

    def setUp(self):
        self.long_text = "This is a very long line of text to test split_text_by_limit function."
        self.limit = len('split_text_by_limit')

    def test_split_text_by_limit(self):
        result = simple_pdf_parsing.split_text_by_limit(self.long_text, self.limit)

        # Check that the result is a list
        self.assertEqual(type(result), list)

        # Check that each line is less than or equal to the limit
        for line in result:
            self.assertLessEqual(len(line), self.limit)

        # Check that all text is included in the output
        self.assertEqual(self.long_text, ' '.join(result))

    def test_empty_text(self):
        result = simple_pdf_parsing.split_text_by_limit("", self.limit)

        # Check that the result is an empty list when text is empty
        self.assertEqual(result, [])

    def test_limit_larger_than_text(self):
        small_text = "short"
        result = simple_pdf_parsing.split_text_by_limit(small_text, self.limit)

        # Check that the result is a list containing the text if the limit is larger than the text
        self.assertEqual(result, [small_text])


if __name__ == '__main__':
    unittest.main()
