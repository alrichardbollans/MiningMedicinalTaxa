import unittest

# Import the method module
from LLM_models.evaluating.evaluation_methods import approximate_match, abbreviated_precise_match, abbreviated_approximate_match, fuzzy_match, \
    precise_match


class TestPreciseMatch(unittest.TestCase):

    def test_both_names_identical(self):
        self.assertTrue(precise_match('John xy', 'John xy'))
        self.assertTrue(precise_match('Johnxy', 'John xy'))

    def test_name1_in_name2(self):
        self.assertFalse(precise_match('John', 'John Doe'))

    def test_start_point_and_punctuation(self):
        self.assertFalse(precise_match('John-Doe', 'JohnDoe'))
        self.assertFalse(precise_match('John-Doe', 'John Doe'))
        self.assertFalse(precise_match('John.Doe', 'John Doe'))
        self.assertFalse(precise_match('John- Doe', 'John Doe'))
        self.assertFalse(precise_match('John - Doe', 'John Doe'))

        self.assertFalse(precise_match('John-Doe', 'JohnDoe', allow_any_start_point=True))
        self.assertFalse(precise_match('John- Doe', 'John Doe', allow_any_start_point=True))
        self.assertFalse(precise_match('John - Doe', 'John Doe', allow_any_start_point=True))

    def test_name2_in_name1(self):
        self.assertFalse(precise_match('John Doe', 'John'))
        self.assertFalse(precise_match('John (Doe', 'John Doe'))
        self.assertFalse(precise_match('John (Doe another few. names', 'John Doe'))
        self.assertFalse(precise_match('John (Doe', 'Doe'))
        self.assertFalse(precise_match('John (Doe another few. names', 'Doe'))

    def test_both_names_diff(self):
        self.assertFalse(precise_match('John', 'Doe'))

    def test_name_in_name_different_case(self):
        self.assertFalse(precise_match('John', 'john'))

    def test_name_in_name_punctuation(self):
        self.assertFalse(precise_match('John', 'John.'))

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

class TestFuzzyMatch(unittest.TestCase):
    
    def symmetric_True(self, a ,b):
        # It's not actually symmetric
        self.assertTrue(fuzzy_match(b, a))
        self.assertTrue(fuzzy_match(a, b))
        
    def symmetric_False(self, a ,b):
        self.assertFalse(fuzzy_match(b, a))
        self.assertFalse(fuzzy_match(a, b))
        
    def test_both_names_identical(self):
        self.symmetric_True('John xy', 'John xy')

    def test_name1_in_name2(self):
        self.symmetric_False('John', 'John Doe')

    def test_start_point_and_punctuation(self):
        self.symmetric_True('John-Doe', 'JohnDoe')
        self.symmetric_True('John- Doe', 'John Doe')
        self.assertTrue('John - Doe', 'John Doe')

    def test_name2_in_name1(self):
        self.symmetric_False('John Doe', 'John')
        self.symmetric_True('John (Doe', 'John Doe')
        self.symmetric_False('melissa officinalis', 'salvia officinalis')
        self.symmetric_False('salvia officinalis', 'melissa officinalis')
        self.symmetric_False('John (Doe another few. names', 'John Doe')
        self.symmetric_False('John (Doe', 'Doe')
        self.symmetric_False('John (Doe another few. names', 'Doe')



    def test_both_names_diff(self):
        self.symmetric_False('John', 'Doe')

    def test_name_in_name_different_case(self):
        self.symmetric_False('John', 'john')

    def test_name_in_name_punctuation(self):
        self.symmetric_True('Johnnnn', 'Johnnnn.')

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
