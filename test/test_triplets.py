import unittest
from bwmd.compressor import load_vectors
from tqdm import tqdm
from bwmd.distance import convert_vectors_to_dict, BWMD
import dill
import time
from nltk.corpus import stopwords
sw = stopwords.words("english")


# TODO: Documentation justifying the scoring method.
# TODO: First-order function as additional param to the
# evaluation method.
N_SAMPLES = 10
MODELS = [
    'glove',
    # 'fasttext',
    # 'word2vec'
]

def load_wikipedia_corpus(n_samples):
    '''
    Load the wikipedia corpus data.

    Parameters
    ----------
        n_samples : int
            Number of samples to retrieve.

    Returns
    ---------
        corpus : list
            List of triplets as tuples.
    '''
    # Load all the triplets and clean.
    corpus = []
    print('Loading wikipedia corpus ...')
    for i in tqdm(range(n_samples)):
        try:
            with open(f'res\\datasets\\triplets_\\wikipedia-{i}', 'rb') as f:
                corpus.append(dill.load(f))

        except FileNotFoundError:
            continue

    return corpus


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
    n_samples = N_SAMPLES
    # Initialize BWMD.
    print('Initializing bwmd object ...')
    bwmd = BWMD(model, dim, with_syntax=syntax,
                    raw_hamming=raw_hamming,
                    full_cache=True)

    # Wikipedia corpus.
    corpus = load_wikipedia_corpus(n_samples)
    start = time.time()
    # List to store the errors.
    score = []
    print('Computing score ...')
    for triplet in tqdm(corpus):
        # Split the texts and remove unattested tokens.
        a, b, c = tuple(map(lambda x: [tok for tok in x if tok in bwmd.vectors], triplet))
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
    return ((end - start) / 60) / n_samples, sum(score) / len(score)


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
            compute_time, score = evaluate_triplets_task(model, '512', True, False)
            print()
            print(model)
            print(str(round(compute_time, 2)), 'minutes/iter')
            print(str(score), 'percent error')
            print()

    def test_wikipedia_tables_no_syntax(self):
        '''
        Evaluations on wikipedia triplets data
        without syntax information.
        '''
        for model in MODELS:
            compute_time, score = evaluate_triplets_task('glove', '512', False, False)
            print()
            print(model)
            print(str(round(compute_time, 2)), 'minutes/iter')
            print(str(score), 'percent error')
            print()

    def test_wikipedia_other_metrics(self):
        '''
        Evaluations on wikipedia triplets task
        for other metrics.
        '''
        def run_test_single_metric(corpus, metric, name):
            '''
            '''
            start = time.time()
            # List to store the errors.
            score = []
            print('Computing score ...')
            for triplet in tqdm(corpus):
                # Split the texts and remove unattested tokens.
                a, b, c = tuple(map(lambda x: [tok for tok in x if \
                                 tok in bwmd.vectors or tok in sw], triplet))
                # Get distances for assessment.
                a_b = metric(a, b)
                a_c = metric(a, c)
                b_c = metric(b, c)
                # Conditional for scoring
                if a_b < a_c and a_b < b_c:
                    # Just use a binary scoring method.
                    score.append(0)
                else:
                    score.append(1)

            end = time.time()
            # Return simple average as percentage error.
            compute_time, score = ((end - start) / 60) / n_samples, sum(score) / len(score)
            # Just print the results.
            print()
            print(name)
            print(str(round(compute_time, 4)), 'minutes/iter')
            print(str(score), 'percent error')
            print()

        n_samples = N_SAMPLES
        # Test WCD, WMD, and RWMD on real-value vectors.
        print('Initializing bwmd object ...')
        bwmd = BWMD('glove', with_syntax=False,
                        raw_hamming=False,
                        full_cache=True)

        # Wikipedia corpus.
        corpus = load_wikipedia_corpus(n_samples)
        # Get the score
        run_test_single_metric(corpus, bwmd.get_wcd, 'Word Centroid Distance')
        run_test_single_metric(corpus, bwmd.get_wmd, "Word Mover's Distance")
        run_test_single_metric(corpus, bwmd.get_rwmd, "Relaxed Word Mover's Distance")

        # Create a new BWMD object for the related distance,
        # because it uses a lookup table rather than the actual distance.
        bwmd = BWMD('glove', '512', with_syntax=False,
                        raw_hamming=False,
                        full_cache=True)
        # Evaluate on the corpus.
        run_test_single_metric(corpus, bwmd.get_relrwmd, "Related Relaxed Word Mover's Distance")
