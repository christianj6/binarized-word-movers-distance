import unittest
from bwmd.compressor import load_vectors
from bwmd.distance import convert_vectors_to_dict, build_partitions_lookup_tables, BWMD


VECTORS = 'glove-512'
REAL_VALUE_PATH = 'glove.txt'
DIM = 512
COMPRESSION = 'bool_'

class TestCase(unittest.TestCase):
    '''
    Test cases for distance module
    including cluster-based lookup table,
    cache-removal policy, and distance calculation
    functionality.
    '''
    def test_partition_lookup_table(self):
        '''
        Test for building the partition lookup table
        accessed during distance-computations in
        combination with intelligent caching policy.
        '''
        vectors_filepath = f"{VECTORS}\\vectors.txtc"
        vectors, words = load_vectors(
            vectors_filepath,
            size=10_000,
            expected_dimensions=DIM,
            expected_dtype=COMPRESSION,
            get_words=True
        )
        vectors = convert_vectors_to_dict(vectors, words)
        token_to_centroid = build_partitions_lookup_tables(
            vectors,
            I=11,
            real_value_path=REAL_VALUE_PATH,
            vector_size=DIM
        )


    def test_bwmd_distance(self):
        '''
        Test for computing the BWMD distance score
        for a small set of test sentences.
        '''
        # Initialize BWMD.
        bwmd = BWMD('glove-512', 512, with_syntax=True, raw_hamming=True)
        # Sample texts for testing.
        text_a = 'Obama speaks to the media in Illinois .'
        text_b = 'The President greets the press in Chicago .'
        # Preprocess the texts.
        text_a, text_b = text_a.split(' '), text_b.split(' ')
        # Compute distance between the texts.
        distance = bwmd.get_distance(text_a, text_b)
        print(distance)


    def test_bwmd_pairwise(self):
        '''
        Test for computing the BWMD distance across
        a corpus of documents and returing a
        pairwise distance matrix.
        '''
        # Initialize BWMD.
        bwmd = BWMD('glove-512', 512, with_syntax=True, raw_hamming=True)
        # Initialize corpus of texts.
        text_a = 'Obama speaks to the media in Illinois'
        text_b = 'The President greets the press in Chicago'
        text_c = 'The man saw the woman'
        text_d = 'The woman saw the man'
        corpus = [text_a, text_b, text_c, text_d]
        # Preprocess the texts.
        corpus = [text.split(' ') for text in corpus]
        # Compute pairwise distances.
        matrix = bwmd.pairwise(corpus)
        print(matrix)


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
        vectors, words = load_vectors(
                vector,
                size=200_000,
                expected_dimensions=int(dim),
                expected_dtype=COMPRESSION,
                get_words=True
            )
        vectors = convert_vectors_to_dict(vectors, words)
        tables = build_partitions_lookup_tables(
                vectors,
                I=11,
                real_value_path=f"{vector[:-4]}.txt",
                vector_size=dim
            )


if __name__ == '__main__':
    compute_all_lookup_tables()
