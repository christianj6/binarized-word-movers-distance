import unittest
from bwmd.tools import load_vectors, convert_vectors_to_dict
from bwmd.distance import BWMD
from tqdm import tqdm
import os
import pickle
import time


def load_wikipedia_corpus(n_samples):
    """
    Load the wikipedia corpus data.

    Parameters
    ----------
        n_samples : int
            Number of samples to retrieve.

    Returns
    ---------
        corpus : list
            List of triplets as tuples.
    """
    # Load all the triplets and clean.
    corpus = []
    print("Loading wikipedia corpus ...")
    for i in tqdm(range(n_samples)):
        try:
            with open(
                os.path.join(os.getcwd(), "test", "data", "triplets", f"wikipedia-{i}"),
                "rb",
            ) as f:
                corpus.append(pickle.load(f))

        except FileNotFoundError:
            continue

    return corpus


def evaluate_triplets_task(model, dim, n_samples):
    """
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
    """
    # Initialize BWMD.
    print("Initializing bwmd object ...")
    bwmd = BWMD(
        model_path=os.path.join(os.getcwd(), "test", "data", model),
        dim=dim,
        size_vocab=1000,
        language="english",
    )
    # Wikipedia corpus.
    corpus = load_wikipedia_corpus(n_samples)
    start = time.time()
    # List to store the errors.
    score = []
    print("Computing score ...")
    for triplet in tqdm(corpus):
        # Split the texts and remove unattested tokens.
        a, b, c = bwmd.preprocess_corpus(triplet)
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


class TestCaseTriplets(unittest.TestCase):
    """
    Test cases for Stanford triplets
    evaluation task, cf Werner (2019).
    """

    MODELS = ["glove"]
    N_SAMPLES = 25

    def test_wikipedia(self):
        for m in self.MODELS:
            compute, score = evaluate_triplets_task(m, 512, self.N_SAMPLES)
            print(f"{m} - {score} - {compute}")
