import unittest

from pre_labelling.evaluating import chunkwise_evaluations


class TestPreciseOutputAnnotationMatch(unittest.TestCase):

    def test_match(self):
        a1 = {"key1": "value1", "key2": "value2"}
        a2 = {"key1": "value1", "key2": "value2"}

        self.assertTrue(chunkwise_evaluations.precise_output_annotation_match(a1, a2))

    def test_no_match(self):
        a1 = {"key1": "value1", "key2": "value2"}
        a2 = {"key1": "value1", "key2": "value3"}

        self.assertFalse(chunkwise_evaluations.precise_output_annotation_match(a1, a2))

    def test_extra_key_in_second_dict(self):
        a1 = {"key1": "value1", "key2": "value2"}
        a2 = {"key1": "value1", "key2": "value2", "key3": "value3"}

        self.assertFalse(chunkwise_evaluations.precise_output_annotation_match(a1, a2))

    def test_different_types(self):
        a1 = {"key1": "value1", "key2": "value2"}
        a2 = {"key1": 1, "key2": 2}

        self.assertFalse(chunkwise_evaluations.precise_output_annotation_match(a1, a2))

class TestApproximateOutputAnnotationMatch(unittest.TestCase):

    def test_approximate_output_annotation_match_matching_attributes(self):
        annotation1 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello',
            'to_text': 'World'
        }
        annotation2 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello there',
            'to_text': 'The world is round'
        }
        self.assertTrue(chunkwise_evaluations.approximate_output_annotation_match(annotation1, annotation2))

        annotation1 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello there',
            'to_text': 'World'
        }
        annotation2 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello ',
            'to_text': 'The world is round'
        }
        self.assertTrue(chunkwise_evaluations.approximate_output_annotation_match(annotation1, annotation2))

    def test_approximate_output_annotation_match_nonmatching_attributes(self):
        annotation1 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello',
            'to_text': 'World'
        }
        annotation2 = {
            'from_label': 'label3',
            'to_label': 'label4',
            'relationship': 'different_relation',
            'from_text': 'Goodbye there',
            'to_text': 'The world is round'
        }
        self.assertFalse(chunkwise_evaluations.approximate_output_annotation_match(annotation1, annotation2))

    def test_approximate_output_annotation_match_no_text_overlap(self):
        annotation1 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Hello',
            'to_text': 'World'
        }
        annotation2 = {
            'from_label': 'label1',
            'to_label': 'label2',
            'relationship': 'relation',
            'from_text': 'Goodbye',
            'to_text': 'Universe'
        }
        self.assertFalse(chunkwise_evaluations.approximate_output_annotation_match(annotation1, annotation2))


if __name__ == '__main__':
    unittest.main()

