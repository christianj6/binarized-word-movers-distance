import unittest
from bwmd.tools import load_vectors, convert_vectors_to_dict
from bwmd.distance import BWMD
import os


class TestCasesDistance(unittest.TestCase):
    """
    Test cases for the distance module.
    """

    MODEL = "glove"
    SIZE = 1_000
    DIM = 512
    COMPRESSION = "bool_"

    def test_bwmd_distance(self):
        """ """
        bwmd = BWMD(
            os.path.join(os.getcwd(), "test", "data", self.MODEL),
            dim=self.DIM,
            size_vocab=self.SIZE,
            language="english",
        )
        # Sample texts for testing.
        text_a = "Obama speaks to the media in Illinois ."
        text_b = "The President greets the press in Chicago ."
        # Preprocess the texts.
        text_a, text_b = tuple(
            map(lambda x: bwmd.preprocess_text(x.split(" ")), (text_a, text_b))
        )
        # Compute distance between the texts.
        distance = bwmd.get_distance(text_a, text_b)
        print(distance)

    def test_bwmd_pairwise(self):
        """
        Test for computing the BWMD distance across
        a corpus of documents and returing a
        pairwise distance matrix.
        """
        bwmd = BWMD(
            os.path.join(os.getcwd(), "test", "data", self.MODEL),
            dim=self.DIM,
            size_vocab=self.SIZE,
            language="english",
        )
        # Initialize corpus of texts.
        text_a = "Obama speaks to the media in Illinois"
        text_b = "The President greets the press in Chicago"
        text_c = "The man saw the woman"
        text_d = "The woman saw the man"
        corpus = [text_a, text_b, text_c, text_d]
        # Preprocess the texts.
        corpus = [text.split(" ") for text in corpus]
        corpus = bwmd.preprocess_corpus(corpus)
        # Compute pairwise distances.
        matrix = bwmd.pairwise(corpus)
        print(matrix)
