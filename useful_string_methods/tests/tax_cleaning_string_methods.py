import unittest

from useful_string_methods import filter_name_list_using_sci_names


class TestExamples(unittest.TestCase):

    def test_duplicates(self):
        taxa = ['Aspidosperma', 'Aspidosperma', 'Cinchona']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['Aspidosperma', 'Aspidosperma', 'Cinchona']

        self.assertEqual(result, expected_output)

    def test_punctuation(self):
        taxa = ['Aspidosperma.', '!Aspidosperma', '!Cinchona. ']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['Aspidosperma.', '!Aspidosperma', '!Cinchona. ']

        self.assertEqual(result, expected_output)

    def test_casing(self):
        taxa = ['aspidosperma.', '!AspidOsperma', '!Cinchona ']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['aspidosperma.', '!AspidOsperma', '!Cinchona ']

        self.assertEqual(result, expected_output)

    def test_species(self):
        taxa = ['hypericum perforatum', '!AspidOsperma', '!Cinchona ', 'hypericum perforatum_depression']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['hypericum perforatum', '!AspidOsperma', '!Cinchona ', 'hypericum perforatum_depression']

        self.assertEqual(result, expected_output)

    def test_hybrids(self):
        taxa = ['× hypericum perforatum', '+ !AspidOsperma', '+ !Cinchona ', 'hypericum × perforatum_depression']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['× hypericum perforatum', '+ !AspidOsperma', '+ !Cinchona ', 'hypericum × perforatum_depression']

        self.assertEqual(result, expected_output)

    def test_abbvs(self):
        taxa = ['× H. beukmanii (C.A.Lückh.) G.D.Rowley', '× H. beukmanii ', 'h. perforatum_depression', 'h. perforatum']
        result = filter_name_list_using_sci_names(taxa)

        expected_output = ['× H. beukmanii (C.A.Lückh.) G.D.Rowley', '× H. beukmanii ', 'h. perforatum']

        self.assertEqual(result, expected_output)
if __name__ == "__main__":
    unittest.main()
