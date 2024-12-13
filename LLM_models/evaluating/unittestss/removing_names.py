# Importing required modules
import unittest

from LLM_models.evaluating.evaluation_methods import TaxaData
from LLM_models.evaluating.evaluation_methods import clean_model_annotations_using_taxonomy_knowledge
from LLM_models.structured_output_schema import Taxon


class TestEvaluationMethods(unittest.TestCase):

    def setUp(self):
        # Create some TaxaData for testing
        self.model_annotations1 = TaxaData(taxa=[{'scientific_name': 'lion'}, {'scientific_name': 'tiger'}, {'scientific_name': 'bear'}])
        self.model_annotations2 = TaxaData(taxa=[{'scientific_name': 'cinchona calisaya', 'medical_conditions':['burning', 'stings']}, {'scientific_name': 'rat'}, {'scientific_name': 'lion'}])
        self.model_annotations2a = TaxaData(taxa=[{'scientific_name': 'c.  calisaya', 'medical_conditions':['burning', 'stings']}, {'scientific_name': 'rat'}, {'scientific_name': 'lion'}])
        self.model_annotations3 = TaxaData(taxa=[])

    def test_clean_model_annotations_using_taxonomy_knowledge(self):
        # Test with model_annotations1
        result = clean_model_annotations_using_taxonomy_knowledge(self.model_annotations1)
        self.assertEqual(result.taxa, [])

        # Test with model_annotations2
        result = clean_model_annotations_using_taxonomy_knowledge(self.model_annotations2)
        self.assertEqual(result.taxa, [Taxon(scientific_name='cinchona calisaya', medical_conditions=['burning', 'stings'], medicinal_effects=None)])
        resulta = clean_model_annotations_using_taxonomy_knowledge(self.model_annotations2)
        self.assertEqual(resulta.taxa, [Taxon(scientific_name='cinchona calisaya', medical_conditions=['burning', 'stings'], medicinal_effects=None)])

        # Test with model_annotations3
        result = clean_model_annotations_using_taxonomy_knowledge(self.model_annotations3)
        self.assertEqual(result.taxa, [])


if __name__ == '__main__':
    unittest.main()
