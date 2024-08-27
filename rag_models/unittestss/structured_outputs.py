import unittest
from rag_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, Taxon, TaxaData, \
    convert_human_annotations_to_taxa_data_schema


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

    def test_empty_input(self):
        self.assertEqual(deduplicate_and_standardise_output_taxa_lists([]), TaxaData(taxa=[]))

    def test_single_element_list(self):
        taxon = Taxon(scientific_name='TestName', medical_conditions=['Condition1'], medicinal_effects=['Effect1'])
        result = deduplicate_and_standardise_output_taxa_lists([taxon])
        self.assertEqual(len(result.taxa), 1)
        self.assertEqual(result.taxa[0].scientific_name, 'testname')
        self.assertEqual(result.taxa[0].medical_conditions, ['condition1'])
        self.assertEqual(result.taxa[0].medicinal_effects, ['effect1'])

    def test_duplicate_scientific_names(self):
        taxon1 = Taxon(scientific_name='TestName', medical_conditions=['Condition1'], medicinal_effects=['Effect1'])
        taxon2 = Taxon(scientific_name='TestName', medical_conditions=['Condition2'], medicinal_effects=['Effect2'])
        result = deduplicate_and_standardise_output_taxa_lists([taxon1, taxon2])
        self.assertEqual(len(result.taxa), 1)
        self.assertEqual(result.taxa[0].scientific_name, 'testname')
        self.assertEqual(sorted(result.taxa[0].medical_conditions), ['condition1', 'condition2'])
        self.assertEqual(sorted(result.taxa[0].medicinal_effects), ['effect1', 'effect2'])

    def test_null_conditions_and_effects(self):
        taxon = Taxon(scientific_name='TestName', medical_conditions=['null'], medicinal_effects=['null'])
        result = deduplicate_and_standardise_output_taxa_lists([taxon])
        self.assertEqual(len(result.taxa), 1)
        self.assertEqual(result.taxa[0].scientific_name, 'testname')
        self.assertEqual(result.taxa[0].medical_conditions, None)
        self.assertEqual(result.taxa[0].medicinal_effects, None)

    def test_clean_strings_called(self):
        taxon = Taxon(scientific_name=' TestName ', medical_conditions=[' Condition1 '], medicinal_effects=[' Effect1 '])
        result = deduplicate_and_standardise_output_taxa_lists([taxon])
        self.assertEqual(len(result.taxa), 1)
        self.assertEqual(result.taxa[0].scientific_name, 'testname')
        self.assertEqual(result.taxa[0].medical_conditions, ['condition1'])
        self.assertEqual(result.taxa[0].medicinal_effects, ['effect1'])



class TestStructuredOutputSchema(unittest.TestCase):

    def test_convert_human_annotations_to_taxa_data_schema(self):

        human_ner_annotations = [{'value': {'label': 'Scientific Fungus Name', 'text': 'Taxon1'}}]
        human_re_annotations = [
            {'from_entity': {'value': {'text': 'Taxon1'}}, 'to_entity': {'value': {'text': 'Effect1'}}, 'label': 'treats_medical_condition'},
            {'from_entity': {'value': {'text': 'Taxon1'}}, 'to_entity': {'value': {'text': 'Condition1'}}, 'label': 'has_medicinal_effect'},
            {'from_entity': {'value': {'text': 'Taxon2'}}, 'to_entity': {'value': {'text': 'Condition1'}}, 'label': 'has_medicinal_effect'}
        ]

        result = convert_human_annotations_to_taxa_data_schema(human_ner_annotations, human_re_annotations)

        self.assertEqual(result.taxa[0].scientific_name, 'Taxon1')
        self.assertEqual(result.taxa[1].scientific_name, 'Taxon2')
        self.assertEqual(result.taxa[0].medical_conditions, ['Effect1'])
        self.assertEqual(result.taxa[0].medicinal_effects, ['Condition1'])
        self.assertEqual(result.taxa[1].medicinal_effects, ['Condition1'])

    def test_convert_human_annotations_incorrect_label_to_taxa_data_schema(self):

        human_ner_annotations = [{'value': {'label': 'TAXON_ENTITY_CLASSES', 'text': 'Taxon1'}}]
        human_re_annotations = [
            {'from_entity': {'value': {'text': 'Taxon1'}}, 'to_entity': {'value': {'text': 'Effect1'}}, 'label': 'incorrect_label'},
        ]

        with self.assertRaises(ValueError):
            convert_human_annotations_to_taxa_data_schema(human_ner_annotations, human_re_annotations)

    def test_convert_human_annotations_no_conditions_or_effects_to_taxa_data_schema(self):

        human_ner_annotations = [{'value': {'label': 'Scientific Plant Name', 'text': 'Taxon1'}}]
        human_re_annotations = []

        result = convert_human_annotations_to_taxa_data_schema(human_ner_annotations, human_re_annotations)

        self.assertEqual(result.taxa[0].scientific_name, 'Taxon1')
        self.assertIsNone(result.taxa[0].medical_conditions)
        self.assertIsNone(result.taxa[0].medicinal_effects)


if __name__ == "__main__":
    unittest.main()
