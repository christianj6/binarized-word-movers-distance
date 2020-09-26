import unittest
from bwmd.compressor import load_vectors
from bwmd.distance import convert_vectors_to_dict, build_kmeans_lookup_tables, BWMD


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
                            size=10_000,
                            expected_dimensions=DIM,
                                expected_dtype=COMPRESSION, get_words=True)
        vectors = convert_vectors_to_dict(vectors, words)
        tables = build_kmeans_lookup_tables(vectors, I=5, path=VECTORS, vector_size=DIM)


    def test_bwmd_distance(self):
        '''
        Test for computing the BWMD similarity score
        for a small set of test sentences.
        '''
        # Initialize BWMD.
        bwmd = BWMD('glove', '512')
        # Sample texts for testing.
        text_a = ['Obama speaks to the media in Illinois.']
        text_b = ['The President greets the press in Chicago.']
        # TODO: Preprocess the texts.
        # TODO: Compute DISTANCE between the texts.
        pass


    def test_bwmd_pairwise(self):
        '''
        '''
        # TODO: Initialize BWMD.
        # TODO: Compute pairwise distances for a
        # list of texts.
        # TODO: Evaluate the cache-removal policy.
        pass


def compute_all_lookup_tables():
    '''
    Automation for computing all the lookup tables
    for all vector models.
    '''
    vectors_to_compute = [
        'glove-256',
        'glove-512',
        'fasttext-256',
        'fasttext-512',
        'word2vec-256',
        'word2vec-512'
    ]
    for vector in vectors_to_compute:
        dim = vector[-3:]
        vector = f"res\\{vector}.txtc"
        vectors, words = load_vectors(
                vector,
                expected_dimensions=int(dim),
                expected_dtype=COMPRESSION,
                get_words=True
            )
        vectors = convert_vectors_to_dict(vectors, words)
        tables = build_kmeans_lookup_tables(
                vectors,
                I=5,
                path=vector,
                vector_size=dim
            )

