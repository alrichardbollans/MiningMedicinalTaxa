import unittest

from rag_models.loading_files import split_text_chunks


class TestLoading(unittest.TestCase):

    def test_split_text_chunks(self):
        words = ['this is some data to chunk haha']
        result = split_text_chunks(words, 5)
        assert result == ['this is some data', 'data to chunk haha']

if __name__ == "__main__":
    unittest.main()
