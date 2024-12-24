import unittest

from useful_string_methods import filter_name_list_with_species_names


class TestABBVExamples(unittest.TestCase):

    def test_duplicates(self):
        taxa = ['A. tomentosum', 'A. tomentosum', 'Aspidosperma tomentosum']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['A. tomentosum', 'A. tomentosum', 'Aspidosperma tomentosum']

        self.assertEqual(result, expected_output)

    def test_punctuation(self):
        taxa = ['A. tomentosum.', 'A. tomentosum']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['A. tomentosum.', 'A. tomentosum']

        self.assertEqual(result, expected_output)

    def test_casing(self):
        taxa = ['a.  tomentosum.', '!AspidOsperma', '!C. calisaya', 'unknown', ', ', '']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['a.  tomentosum.', '!C. calisaya']

        self.assertEqual(result, expected_output)

    def test_species(self):
        taxa = ['h. perforatum', '!AspidOsperma', '!Cinchona ', 'h. perforatum depression', 'Aspidosperma paniculata', 'Aspidosperma tomentosum.',
                'C. calisaya depression']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['h. perforatum', 'h. perforatum depression', 'C. calisaya depression', 'Aspidosperma tomentosum.']

        self.assertEqual(sorted(result), sorted(expected_output))

    def test_hybrids(self):
        taxa = ['+ a. delponii', '+ !AspidOsperma', '+ !Cinchona ', 'h. × perforatum depression', 'h. × jusbertii', '+ amygdalopersica formontii',
                '+ amygdalopersica formontii depression', 'acer × jakelyanum']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['+ a. delponii', 'h. × jusbertii', '+ amygdalopersica formontii', '+ amygdalopersica formontii depression',
                           'acer × jakelyanum']

        self.assertEqual(result, expected_output)

    def test_abbvs(self):
        taxa = ['× H. beukmanii (C.A.Lückh.) G.D.Rowley', '× H. beukmanii ', 'h. perforatum_depression', 'h. perforatum']
        result = filter_name_list_with_species_names(taxa)

        expected_output = ['× H. beukmanii (C.A.Lückh.) G.D.Rowley', '× H. beukmanii ', 'h. perforatum']

        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
