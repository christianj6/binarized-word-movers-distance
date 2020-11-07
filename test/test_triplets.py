import unittest
from bwmd.compressor import load_vectors
from tqdm import tqdm
from bwmd.distance import convert_vectors_to_dict, BWMD
import dill
import time


# TODO: Fix issue that some dependencies are not accounted for.
# TODO: Documentation justifying the scoring method.
# TODO: First-order function as additional param to the
# evaluation method.

MODELS = [
    'glove',
    'fasttext',
    'word2vec'
]

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
    # Initialize BWMD.
    print('Initializing bwmd object ...')
    bwmd = BWMD(model, dim, with_syntax=syntax,
                    raw_hamming=raw_hamming,
                    full_cache=True)

    # Load all the triplets and clean.
    corpus = []
    print('Loading wikipedia corpus ...')
    for i in tqdm(range(100)):
        try:
            with open(f'res\\datasets\\triplets_\\wikipedia-{i}', 'rb') as f:
                corpus.append(dill.load(f))

        except FileNotFoundError:
            continue

    start = time.time()
    # List to store the errors.
    score = []
    print('Computing score ...')
    for triplet in tqdm(corpus):
        # Split the texts.
        a, b, c = triplet
        # Get distances for assessment.
        a_b = bwmd.get_distance(a, b)
        a_c = bwmd.get_distance(a, c)
        b_c = bwmd.get_distance(b, c)
        # Conditional for scoring
        if a_b < a_c and a_b < b_c:
            # Just use a binary scoring method.
            score.append(0)
        else:
            score.append(1)

    end = time.time()
    # Return simple average as percentage error.
    return (end - start) / 60, sum(score) / len(score)


class TestCase(unittest.TestCase):
    '''
    Test cases for Stanford triplets
    evaluation task, cf Werner (2019).
    '''
    def test_wikipedia_tables_syntax(self):
        '''
        Evaluations on wikipedia triplets data
        with syntax information.
        '''
        for model in MODELS:
            time, score = evaluate_triplets_task(model, '512', True, False)
            print(model)
            print(str(round(time, 2)), 'minutes')
            print(str(score), 'percent error')
            print()

    def test_wikipedia_tables_no_syntax(self):
        '''
        Evaluations on wikipedia triplets data
        without syntax information.
        '''
        for model in MODELS:
            time, score = evaluate_triplets_task('glove', '512', False, False)
            print(model)
            print(str(round(time, 2)), 'minutes')
            print(str(score), 'percent error')
            print()

    def test_wikipedia_other_metrics(self):
        '''
        Evaluations on wikipedia triplets task
        for other metrics.
        '''
        pass
