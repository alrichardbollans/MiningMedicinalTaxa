import unittest

# Import the method module
from rag_models.evaluating.evaluation_methods import approximate_match, abbreviated_precise_match, abbreviated_approximate_match


class TestApproximateMatch(unittest.TestCase):

    def test_both_names_identical(self):
        self.assertTrue(approximate_match('John xy', 'John xy'))

    def test_name1_in_name2(self):
        self.assertTrue(approximate_match('John', 'John Doe'))

    def test_start_point_and_punctuation(self):
        self.assertFalse(approximate_match('John-Doe', 'JohnDoe'))
        self.assertTrue(approximate_match('John- Doe', 'John Doe'))
        self.assertTrue(approximate_match('John - Doe', 'John Doe'))

        self.assertTrue(approximate_match('John-Doe', 'JohnDoe', allow_any_start_point=True))
        self.assertTrue(approximate_match('John- Doe', 'John Doe', allow_any_start_point=True))
        self.assertTrue(approximate_match('John - Doe', 'John Doe', allow_any_start_point=True))

    def test_name2_in_name1(self):
        self.assertTrue(approximate_match('John Doe', 'John'))
        self.assertTrue(approximate_match('John (Doe', 'John Doe'))
        self.assertTrue(approximate_match('John (Doe another few. names', 'John Doe'))
        self.assertFalse(approximate_match('John (Doe', 'Doe'))
        self.assertFalse(approximate_match('John (Doe another few. names', 'Doe'))

    def test_both_names_diff(self):
        self.assertFalse(approximate_match('John', 'Doe'))

    def test_name_in_name_different_case(self):
        self.assertFalse(approximate_match('John', 'john'))

    def test_name_in_name_punctuation(self):
        self.assertTrue(approximate_match('John', 'John.'))


class TestEvaluationMethods(unittest.TestCase):
    def test_abbreviated_precise_match(self):
        name1 = "Python"
        name2 = "Python"
        self.assertTrue(abbreviated_precise_match(name1, name2))
        self.assertTrue(abbreviated_approximate_match(name1, name2))

        name1 = "Python agb x"
        name2 = "P. agb x"
        self.assertTrue(abbreviated_precise_match(name1, name2))
        self.assertTrue(abbreviated_approximate_match(name1, name2))

        with self.assertRaises(ValueError):
            self.assertTrue(abbreviated_approximate_match(name1, '   ' + name2))

        with self.assertRaises(ValueError):
            self.assertTrue(abbreviated_approximate_match('   ' + name1, name2))

        self.assertTrue(abbreviated_precise_match(name1, name1))
        self.assertTrue(abbreviated_approximate_match(name1, name1))

        name1 = "Python"
        name2 = "PYT"
        self.assertFalse(abbreviated_precise_match(name1, name2))

        name1 = "PYT"
        name2 = "Python"
        self.assertFalse(abbreviated_precise_match(name1, name2))

        name1 = "Java"
        name2 = "Python"
        self.assertFalse(abbreviated_precise_match(name1, name2))

        name1 = "JaVa"
        name2 = "JAV"
        self.assertFalse(abbreviated_precise_match(name1, name2))


class TestAbbreviatedApproximateMatch(unittest.TestCase):

    def test_abbreviated_approximate_match_exact(self):
        self.assertTrue(abbreviated_approximate_match("test", "test"))

    def test_abbreviated_approximate_match_differs_case(self):
        self.assertFalse(abbreviated_approximate_match("Test", "test"))

    def test_abbreviated_approximate_match_differslength_abbrev(self):
        self.assertFalse(abbreviated_approximate_match("t", "test"))

    def test_abbreviated_approximate_match_differs_length(self):
        self.assertFalse(abbreviated_approximate_match("tst", "test"))

    def test_abbreviated_approximate_match_abbrev_switched(self):
        self.assertTrue(abbreviated_approximate_match("test one two three", "t. one"))
        self.assertTrue(abbreviated_approximate_match("test one two three", "t. one two"))
        self.assertTrue(abbreviated_approximate_match("test one two three", "t. one two three"))

    def test_abbreviated_approximate_match_negative_case(self):
        self.assertFalse(abbreviated_approximate_match("name", "test"))
        self.assertFalse(abbreviated_approximate_match("test one two three", "one two three"))
        self.assertFalse(abbreviated_approximate_match("test one two three", "o. two three"))
        self.assertFalse(abbreviated_approximate_match("test one two three", "t. two three"))


if __name__ == '__main__':
    unittest.main()
