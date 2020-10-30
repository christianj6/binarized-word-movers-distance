import unittest
from bwmd.compressor import load_vectors
from tqdm import tqdm
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
    import time
    # Initialize BWMD.
    print('Initializing bwmd object ...')
    bwmd = BWMD(model, dim, with_syntax=syntax,
                    raw_hamming=raw_hamming,
                    full_cache=True)

    # Load all the triplets and clean.
    corpus = []
    print('Loading wikipedia corpus ...')
    # for i in tqdm(range(20_000)):
    for i in tqdm(range(20_000)):
        try:
            with open(f'res\\datasets\\triplets\\wikipedia-{i}', 'rb') as f:
                triplet = dill.load(f)
                # Split the texts.
                a_b_c = list(map(lambda x: x.split(), triplet))
                # Clean each tuple.
                new_tuple = []
                for item in a_b_c:
                    # Clean item.
                    new_tuple.append([word.lower() for word in item if word in bwmd.words])
                corpus.append(tuple(new_tuple))

        except FileNotFoundError:
            continue

    times = []
    # List to store the errors.
    score = []
    print('Computing score ...')
    # for triplet in tqdm(corpus):
    for triplet in corpus:
        start = time.time()
        # Split the texts.
        a, b, c = triplet
        # Get distances for assessment.
        a_b = bwmd.get_distance(a, b)
        a_c = bwmd.get_distance(a, c)
        b_c = bwmd.get_distance(b, c)
        # Conditional for scoring
        if a_b < a_c and a_b < b_c:
            print(a_b, a_c, b_c)
            # Just use a binary scoring method.
            score.append(0)
        else:
            score.append(1)

        end = time.time()
        times.append((end - start))
        # print('\n\n')

    print(sum(times) / len(times))

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
        # TODO: Fix issue that some dependencies are not accounted for.
        # TODO: Figure out why so slow and speed up.
        # TODO: Time computations.
        score = evaluate_triplets_task('glove', '512', False, False)
        print(score)

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
