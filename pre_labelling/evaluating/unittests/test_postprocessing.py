import unittest

from pre_labelling.evaluating import post_processing_methods


class TestPostWhitespace(unittest.TestCase):
    def test_leading_trailing_whitespace_none_input(self):
        none_input = None
        self.assertEqual(post_processing_methods.leading_trailing_whitespace(none_input), None)

    def test_leading_trailing_whitespace_empty_string(self):
        empty_string = ""
        self.assertEqual(post_processing_methods.leading_trailing_whitespace(empty_string), empty_string)

    def test_leading_trailing_whitespace_whitespace_only(self):
        whitespace_only = "   "
        self.assertEqual(post_processing_methods.leading_trailing_whitespace(whitespace_only), "")

    def test_leading_trailing_whitespace_leading_trailing_spaces(self):
        leading_trailing_spaces = "  hello world  "
        self.assertEqual(post_processing_methods.leading_trailing_whitespace(leading_trailing_spaces), "hello world")

    def test_leading_trailing_whitespace_no_whitespace(self):
        no_whitespace = "helloworld"
        self.assertEqual(post_processing_methods.leading_trailing_whitespace(no_whitespace), no_whitespace)

class TestPuncatuation(unittest.TestCase):
    def test_leading_trailing_punctuation(self):
        self.assertEqual(post_processing_methods.leading_trailing_punctuation("!Hello, World!"), "Hello, World")
        self.assertEqual(post_processing_methods.leading_trailing_punctuation("!123#"), "123")
        self.assertEqual(post_processing_methods.leading_trailing_punctuation("     Hello     "), "     Hello     ")
        self.assertEqual(post_processing_methods.leading_trailing_punctuation(None), None)
        self.assertEqual(post_processing_methods.leading_trailing_punctuation("#!Python is cool!#"), "Python is cool")
        self.assertEqual(post_processing_methods.leading_trailing_punctuation(".Hello."), "Hello")


class LowercaseTest(unittest.TestCase):
    def test_lowercase_string(self):
        self.assertEqual(post_processing_methods.lowercase('HELLO'), 'hello')

    def test_uppercase_string(self):
        self.assertEqual(post_processing_methods.lowercase('HELLO WORLD'), 'hello world')

    def test_mixed_case_string(self):
        self.assertEqual(post_processing_methods.lowercase('HeLLo WoRLD'), 'hello world')

    def test_integer_input(self):
        # Expects the integer '1234' to raise AttributeError and return input value
        self.assertEqual(post_processing_methods.lowercase(1234), 1234)

    def test_none_value(self):
        self.assertIsNone(post_processing_methods.lowercase(None), None)


class TestPostProcessingMethods(unittest.TestCase):
    def test_clean_strings(self):
        self.assertEqual(post_processing_methods.clean_strings("Hello World!! "), "hello world")
        self.assertEqual(post_processing_methods.clean_strings(" PYTHON "), "python")
        self.assertEqual(post_processing_methods.clean_strings("!AI_Assistant!?"), "ai_assistant")
        self.assertEqual(post_processing_methods.clean_strings("!AI_Assistant!? ."), "ai_assistant")
        self.assertEqual(post_processing_methods.clean_strings("!AI_Assistant!? "), "ai_assistant")


if __name__ == "__main__":
    unittest.main()
