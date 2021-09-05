import unittest
from bwmd.tools import load_vectors, convert_vectors_to_dict
from bwmd.partition import build_partitions_lookup_tables
import os


class TestCasesPartitions(unittest.TestCase):
    """
    Test cases for the partitioning algorithm.
    """

    MODEL = "glove"
    SIZE = 1_000
    DIM = 512
    COMPRESSION = "bool_"

    def test_build_partitions_lookup_table(self):
        """
        Tests that the partitioning
        algorithm works without producing errors.
        """
        fp = os.path.join(os.getcwd(), "test", "data", self.MODEL, "vectors.txtc")
        vectors, words = load_vectors(
            path=fp,
            size=self.SIZE,
            expected_dimensions=self.DIM,
            expected_dtype=self.COMPRESSION,
        )
        vectors_compressed = convert_vectors_to_dict(vectors, words)
        token_to_centroid = build_partitions_lookup_tables(
            vectors_compressed,
            I=6,
            real_value_path=os.path.join(os.getcwd(), "test", "data", f"{self.MODEL}.txt"),
            vector_size=self.DIM,
        )
