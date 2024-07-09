import unittest
import sys

sys.path.append('..')
from testing.evaluation_methods import check_human_annotations, TAXON_ENTITY_CLASSES, MEDICINAL_CLASSES, MEDICINAL_RELATIONS


class TestCheckHumanAnnotations(unittest.TestCase):
    def setUp(self):
        self.valid_human_ner_annotation = [{'value': {'label': TAXON_ENTITY_CLASSES[0], 'text': 'text1'}}]
        self.valid_human_re_annotation = [{'from_entity': {'value': {'labels': [TAXON_ENTITY_CLASSES[0]], 'text': 'text1'}},
                                'to_entity': {'value': {'labels': [MEDICINAL_CLASSES[0]], 'text': 'text2'}}, 'label': MEDICINAL_RELATIONS[0]}]

    def test_check_human_annotation_valid(self):

        try:
            check_human_annotations(self.valid_human_ner_annotation, self.valid_human_re_annotation)
        except ValueError as v:
            self.fail(f"check_human_annotations() raised ValueError unexpectedly: {v}")

    def test_check_human_annotation_invalid_ner_label(self):
        human_ner_annotation = [{'value': {'label': 'invalid_label', 'text': 'text1'}}]
        with self.assertRaises(ValueError):
            check_human_annotations(human_ner_annotation, self.valid_human_re_annotation)

    def test_check_human_annotation_invalid_relation_label(self):
        human_re_annotation = [{'from_entity': {'value': {'labels': ['tax_label1'], 'text': 'text1'}},
                                'to_entity': {'value': {'labels': ['med_label1'], 'text': 'text2'}}, 'label': 'invalid_relation'}]
        with self.assertRaises(ValueError):
            check_human_annotations(self.valid_human_ner_annotation, human_re_annotation)

    def test_check_human_annotation_invalid_labels_count(self):
        human_re_annotation = [{'from_entity': {'value': {'labels': ['tax_label1', 'tax_label2'], 'text': 'text1'}},
                                'to_entity': {'value': {'labels': ['med_label1', 'med_label2'], 'text': 'text2'}}, 'label': 'relation1'}]
        with self.assertRaises(ValueError):
            check_human_annotations(self.valid_human_ner_annotation, human_re_annotation)

    def test_check_human_annotation_invalid_ner_without_taxon(self):
        human_ner_annotation = [{'value': {'label': 'tax_label1', 'text': 'text3'}}]
        with self.assertRaises(ValueError):
            check_human_annotations(human_ner_annotation, self.valid_human_re_annotation)


if __name__ == "__main__":
    unittest.main()
