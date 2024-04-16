import unittest

import numpy as np

from testing.evaluation_methods import NER_metrics

class TestPreciseEntityMatch(unittest.TestCase):
    def setUp(self):
        self.entity1 = {
            "start": 0,
            "end": 5,
            "text": "Hello"
        }
        self.entity2 = {
            "start": 0,
            "end": 5,
            "text": "hello"
        }
        self.entity3 = {
            "start": 0,
            "end": 5,
            "text": "world"
        }
        self.entity4 = {
            "start": 5,
            "end": 10,
            "text": "hello"
        }

    def test_precise_entity_match(self):
        # Test that function returns True for precisely matching entities
        self.assertTrue(NER_metrics.precise_entity_match(self.entity1, self.entity2))

        # Test that function returns False if the text doesn't match
        self.assertFalse(NER_metrics.precise_entity_match(self.entity1, self.entity3))

        # Test that function returns False if the 'start' or 'end' values don't match
        self.assertFalse(NER_metrics.precise_entity_match(self.entity1, self.entity4))


class TestPreciseNERAnnotationMatch(unittest.TestCase):

    def test_precise_NER_annotation_match_same(self):
        a1 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        a2 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        self.assertTrue(NER_metrics.precise_NER_annotation_match(a1, a2))

    def test_precise_NER_annotation_match_different_positions(self):
        a1 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        a2 = {"value": {"start": 1, "end": 3, "label": "ORG", "text": "Google"}}
        self.assertFalse(NER_metrics.precise_NER_annotation_match(a1, a2))

    def test_precise_NER_annotation_match_different_labels(self):
        a1 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        a2 = {"value": {"start": 0, "end": 2, "label": "PER", "text": "Google"}}
        self.assertFalse(NER_metrics.precise_NER_annotation_match(a1, a2))

    def test_precise_NER_annotation_match_different_texts(self):
        a1 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        a2 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Apple"}}
        self.assertFalse(NER_metrics.precise_NER_annotation_match(a1, a2))

    def test_precise_NER_annotation_match_case_insensitive(self):
        a1 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "Google"}}
        a2 = {"value": {"start": 0, "end": 2, "label": "ORG", "text": "google"}}
        self.assertTrue(NER_metrics.precise_NER_annotation_match(a1, a2))


class TestApproximateEntityMatch(unittest.TestCase):
    def test_approximate_match_same_entity(self):
        entity = {"start": 0, "end": 5, "text": "Hello"}
        self.assertTrue(NER_metrics.approximate_entity_match(entity, entity))

    def test_approximate_match_contained_entity(self):
        entity1 = {"start": 0, "end": 10, "text": "Hello World"}
        entity2 = {"start": 0, "end": 5, "text": "Hello"}
        self.assertTrue(NER_metrics.approximate_entity_match(entity1, entity2))
        self.assertTrue(NER_metrics.approximate_entity_match(entity2, entity1))

    def test_approximate_match__contained_entity(self):
        entity1 = {"start": 0, "end": 5, "text": "Hello"}
        entity2 = {"start": 0, "end": 10, "text": "Hello World"}
        self.assertTrue(NER_metrics.approximate_entity_match(entity1, entity2))

    def test_approximate_match_overlap_entity(self):
        entity1 = {"start": 0, "end": 7, "text": "Hello Wo"}
        entity2 = {"start": 5, "end": 11, "text": "World"}
        self.assertFalse(NER_metrics.approximate_entity_match(entity1, entity2))
        self.assertFalse(NER_metrics.approximate_entity_match(entity2, entity1))

    def test_approximate_match_same_start_diff_end_entity(self):
        entity1 = {"start": 0, "end": 5, "text": "Hello"}
        entity2 = {"start": 0, "end": 3, "text": "Hel"}
        self.assertTrue(NER_metrics.approximate_entity_match(entity1, entity2))
        self.assertTrue(NER_metrics.approximate_entity_match(entity2, entity1))

    def test_approximate_match_same_end_diff_start_entity(self):
        entity1 = {"start": 0, "end": 5, "text": "Hello"}
        entity2 = {"start": 2, "end": 5, "text": "llo"}
        self.assertTrue(NER_metrics.approximate_entity_match(entity1, entity2))
        self.assertTrue(NER_metrics.approximate_entity_match(entity2, entity1))

class TestApproximateNER(unittest.TestCase):
    def test_approximate_NER_annotation_match_nonmatching_labels(self):
        entity1 = {"value": {"start": 0, "end": 5, "text": "Hello", "label": "Animal"}}
        entity2 = {"value": {"start": 0, "end": 3, "text": "Hel", "label": "Plant"}}
        self.assertFalse(NER_metrics.approximate_NER_annotation_match(entity1, entity2))
        self.assertFalse(NER_metrics.approximate_NER_annotation_match(entity2, entity1))

    def test_approximate_NER_a1(self):
        entity1 = {"value": {"start": 0, "end": 5, "text": "Hello", "label": "Plant"}}
        entity2 = {"value": {"start": 0, "end": 3, "text": "Hel", "label": "Plant"}}
        self.assertTrue(NER_metrics.approximate_NER_annotation_match(entity1, entity2))
        self.assertTrue(NER_metrics.approximate_NER_annotation_match(entity2, entity1))
        self.assertTrue(NER_metrics.approximate_NER_annotation_match(entity2, entity2))
        self.assertTrue(NER_metrics.approximate_NER_annotation_match(entity1, entity1))


class TestNERMetrics(unittest.TestCase):

    def test_get_metrics_with_empty_inputs(self):
        tp, fp, fn = [], [], []
        precision, recall, f1_score = NER_metrics.get_metrics_from_tp_fp_fn(tp, fp, fn)
        self.assertTrue(np.isnan(precision))
        self.assertTrue(np.isnan(recall))
        self.assertTrue(np.isnan(f1_score))

    def test_get_metrics_with_non_empty_inputs(self):
        tp = [1]*50
        fp = [0]*30
        fn = [0]*20
        precision, recall, f1_score = NER_metrics.get_metrics_from_tp_fp_fn(tp, fp, fn)
        self.assertEqual(precision, len(tp) / (len(tp) + len(fp)))
        self.assertEqual(precision,0.625)
        self.assertEqual(recall, len(tp) / (len(tp) + len(fn)))
        self.assertEqual(recall, 0.7142857142857143)
        self.assertEqual(f1_score, 2*len(tp) / ((2*len(tp)) + len(fp) + len(fn)))
        self.assertEqual(f1_score, 0.6666666666666666)

    def test_get_metrics_with_zero_positives(self):
        tp = []
        fp = []
        fn = []
        precision, recall, f1_score = NER_metrics.get_metrics_from_tp_fp_fn(tp, fp, fn)
        self.assertTrue(np.isnan(precision))
        self.assertTrue(np.isnan(recall))
        self.assertTrue(np.isnan(f1_score))

    def test_get_metrics_with_zero_true_positives(self):
        tp = []
        fp = []
        fn = [1]*20
        precision, recall, f1_score = NER_metrics.get_metrics_from_tp_fp_fn(tp, fp, fn)
        self.assertTrue(np.isnan(precision))
        self.assertEqual(recall, 0)
        self.assertEqual(f1_score, 0)

if __name__ == '__main__':
    unittest.main()
