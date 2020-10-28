import unittest
from bwmd.compressor import load_vectors
from bwmd.distance import convert_vectors_to_dict, BWMD
import dill


# TODO: Documentation justifying the scoring method.

def evaluate_triplets_task(model, dim, syntax, raw_hamming):
    '''
    Score a given model on the wikipedia triplets,
    returning the test error as a percentage.

    Parameters
    ---------
        model : str
            Name of model.
        dim : str
            Dimension of model.
        syntax : bool
            Whether to use syntax
            info in the evaluation.
        raw_hamming : bool
            Whether to just use
            the raw hamming distances for the
            calculations.

    Returns
    ---------
        error : float
            Test error as percentage
            of tuples for which the incorrect
            evaluation was made.
    '''
    # Load all the triplets.
    corpus = []
    for i in range(20_000):
        try:
            with open(f'res\\datasets\\triplets\\wikipedia-{i}', 'rb') as f:
                corpus.append(dill.load(f))

        except FileNotFoundError:
            continue

    # Initialize BWMD.
    bwmd = BWMD(model, dim, with_syntax=syntax,
                    raw_hamming=raw_hamming)

    # List to store the errors.
    score = []
    for triplet in corpus:
        # Split the texts.
        triplet = list(map(lambda x: x.split(), triplet))
        # Pairwise distances.
        matrix = bwmd.pairwise(triplet)
        # Conditional for scoring
        if not matrix[0][1] < matrix[0][2] and matrix[1][0] < matrix[1][2]:
            # Just use a binary scoring method.
            score.append(0)
        else:
            score.append(1)

    # Return simple average as percentage error.
    return sum(score) / len(score)


class TestCase(unittest.TestCase):
    '''
    Test cases for Stanford triplets
    evaluation task, cf Werner (2019).
    '''
    def test_wikipedia_hamming_syntax(self):
        '''
        '''
        # TODO: All test cases must use all embeddings.
        pass

    def test_wikipedia_hamming_no_syntax(self):
        '''
        '''
        pass

    def test_wikipedia_tables_syntax(self):
        '''
        '''
        pass

    def test_wikipedia_tables_no_syntax(self):
        '''
        '''
        pass
