import unittest
from bwmd.compressor import load_vectors
from bwmd.distance import convert_vectors_to_dict, build_kmeans_lookup_tables


VECTORS = 'res\\glove-512.txtc'
DIM = 512
COMPRESSION = 'bool_'

class TestCase(unittest.TestCase):
    '''
    Test cases for distance module
    including cluster-based lookup table,
    cache-removal policy, and distance calculation
    functionality.
    '''
    def test_knn_lookup_table(self):
        '''
        Test for building the knn lookup table
        accessed during distance-computations in
        combination with intelligent caching policy.
        '''
        vectors, words = load_vectors(VECTORS,
                            size=200_000,               # only for prelim tests
                            expected_dimensions=DIM,
                                expected_dtype=COMPRESSION, get_words=True)
        vectors = convert_vectors_to_dict(vectors, words)
        tables = build_kmeans_lookup_tables(vectors, I=5, path=VECTORS)


    def test_bwmd_similarity(self):
        '''
        '''
        # TODO: Initialize BWMD.
        # TODO: Compute distance between two sample texts.
        pass


    def test_bwmd_pairwise(self):
        '''
        '''
        # TODO: Initialize BWMD.
        # TODO: Compute pairwise distances for a
        # list of texts.
        # TODO: Evaluate the cache-removal policy.
        pass
