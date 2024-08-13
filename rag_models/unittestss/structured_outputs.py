import unittest
from rag_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, Taxon, TaxaData

class TestDeduplicateAndStandardiseOutputTaxaLists(unittest.TestCase):

    def test_with_no_duplicate_taxa(self):
        taxa = [Taxon(scientific_name='Taxon1', medical_conditions=['Condition1'], medicinal_effects=['Effect1']),
                Taxon(scientific_name='Taxon2', medical_conditions=['Condition2'], medicinal_effects=['Effect2'])]
        result = deduplicate_and_standardise_output_taxa_lists(taxa)

        expected_output = TaxaData(taxa=[Taxon(scientific_name='taxon1', medical_conditions=['condition1'], medicinal_effects=['effect1']),
                Taxon(scientific_name='taxon2', medical_conditions=['condition2'], medicinal_effects=['effect2'])])

        self.assertEqual(result, expected_output)

    def test_with_duplicate_taxa(self):
        taxa = [Taxon(scientific_name='Taxon1', medical_conditions=['Condition1!'], medicinal_effects=['Effect1']),
                Taxon(scientific_name='Taxon1', medical_conditions=['Condition1 .'], medicinal_effects=['Effect1 '])]
        result = deduplicate_and_standardise_output_taxa_lists(taxa)

        expected_output = TaxaData(taxa=[Taxon(scientific_name='taxon1', medical_conditions=['condition1'], medicinal_effects=['effect1'])])

        self.assertEqual(result, expected_output)

    def test_with_null_fields_in_taxa(self):
        taxa = [Taxon(scientific_name=None, medical_conditions=None, medicinal_effects=None),
                Taxon(scientific_name='Taxon1', medical_conditions=['Condition1'], medicinal_effects=['Effect1'])]
        result = deduplicate_and_standardise_output_taxa_lists(taxa)

        expected_output = TaxaData(taxa=[Taxon(scientific_name='taxon1', medical_conditions=['condition1'], medicinal_effects=['effect1'])])

        self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()
