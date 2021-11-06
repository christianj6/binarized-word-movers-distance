import unittest
from bwmd.tools import load_vectors, convert_vectors_to_dict
from bwmd.compressor import Compressor
from bwmd.partition import build_partitions_lookup_tables
from bwmd.distance import BWMD, WMD, RWMD, RelRWMD, WCD, BOW
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
                os.path.join(
                    os.getcwd(), "test", "data", "triplets", f"wikipedia-{i}"
                ),
                "rb",
            ) as f:
                corpus.append(pickle.load(f))

        except FileNotFoundError:
            continue

    return corpus


def get_empty_results_dict(metrics: list) -> dict:
    """
    Returns an empty data structure intended to
    store test error and compute time for
    the different evaluation metrics.

    Parameters
    ---------
        metrics : list[str]
            List of metric names as strings

    Returns
    ---------
        results : dict[str, dict]
            Empty results dict organizing
            online, offline compute and test
            error.
    """
    results = {
        "time_online": {m: 0 for m in metrics},
        "time_offline": {m: 0 for m in metrics},
        "test_error": {m: 0 for m in metrics},
    }

    return results


def get_encoded_vectors(
    fp: str, original_dimensions: int, reduced_dimensions: int
) -> dict:
    """
    .
    """
    # fit autoencder
    vectors_original, words_original = load_vectors(
        path=fp,
        size=300_000,
        expected_dimensions=original_dimensions,
        skip_first_line=True,
    )
    compressor = Compressor(
        original_dimensions=original_dimensions,
        reduced_dimensions=reduced_dimensions,
        compression="bool_",
    )
    compressor.fit(vectors_original, epochs=10)
    # save encoded vectors
    output_dir = compressor.transform(fp, save=True, n_vectors=200_000)
    # create lookup tables for the exported model
    vectors_encoded, words = load_vectors(
        path=f"{output_dir}\\vectors.txtc",
        size=200_000,
        expected_dimensions=reduced_dimensions,
        expected_dtype="bool_",
    )
    vectors_encoded_dict = convert_vectors_to_dict(vectors_encoded, words)
    vectors_original_dict = convert_vectors_to_dict(
        vectors_original, words_original
    )

    return vectors_encoded_dict, vectors_original_dict


def evaluate_triplets_task(
    model_name: str,
    model_filepath: str,
    input_dim: int,
    encoded_dim: int,
    n_samples: int,
) -> dict:
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
    # organize eval metrics
    metrics = ["wmd", "wcd", "rwmd", "relrwmd", "bow", "bow-l1", "bwmd"]
    # organize corresponding objects
    objects = [WMD, WCD, RWMD, RelRWMD, BOW, BOW, BWMD]
    # organize results in hash table
    results = get_empty_results_dict(metrics)
    # start timer
    t = time.time()
    # get encoded vectors, tracking offline compute time
    vectors_encoded_dict, vectors_original_dict = get_encoded_vectors(
        model_filepath, input_dim, encoded_dim
    )
    # stop timer add to bwmd
    results["time_offline"]["bwmd"] += time.time() - t
    # start timer
    t = time.time()
    # get lookup tables
    lookup_table_path = build_partitions_lookup_tables(
        vectors_encoded_dict,
        I=11,
        real_value_path=model_filepath,
        vector_dim=encoded_dim,
    )
    # we use the same lookup tables for bwmd and rel-rwmd,
    # thus this offline computation time is added to both metrics
    a = time.time() - t
    results["time_offline"]["bwmd"] += a
    results["time_offline"]["relrwmd"] += a
    # Prepare Wikipedia corpus.
    corpus = load_wikipedia_corpus(n_samples)
    # joined corpus needed for tfidf vectorizer
    corpus_joined = [" ".join(doc) for triplet in corpus for doc in triplet]
    # organize parameters for distance objects
    params = [
        {"language": "english", "vectors": vectors_original_dict},  # wmd
        {"language": "english", "vectors": vectors_original_dict},  # wcd
        {"language": "english", "vectors": vectors_original_dict},  # rwmd
        {"language": "english", "cache_path": lookup_table_path},  # relrwmd
        {
            "l1_norm": False,
            "language": "english",
            "corpus": corpus_joined,
        },  # bow
        {
            "l1_norm": True,
            "language": "english",
            "corpus": corpus_joined,
        },  # bow-l1
        {
            "model_path": lookup_table_path,
            "dim": encoded_dim,
            "size_vocab": 200_000,
            "language": "english",
        },  # bwmd
    ]

    for i, (m, o, p) in enumerate(zip(metrics, objects, params)):
        # start timer
        t = time.time()
        # build object and store
        obj = o(**p)
        objects[i] = obj
        # stop timer update results
        results["time_offline"][m] += time.time() - t

    for obj, name in zip(objects, metrics):
        start = time.time()
        # List to store the errors.
        score = []
        for triplet in tqdm(corpus):
            # Split the texts and remove unattested tokens.
            a, b, c = obj.preprocess_corpus(triplet)
            # Get distances for assessment.
            a_b = obj.get_distance(a, b)
            a_c = obj.get_distance(a, c)
            b_c = obj.get_distance(b, c)
            # Conditional for scoring
            if a_b < a_c and a_b < b_c:
                # Just use a binary scoring method.
                score.append(0)

            else:
                score.append(1)

        end = time.time()
        # update online computations
        results["time_online"][name] = (end - start) / n_samples
        # update score; simple average of correct/incorrect triplets
        results["test_error"][name] = sum(score) / len(score)

    return results


def print_results(results: dict) -> None:
    """
    Parses and prints results to stdout.
    """
    print()
    print("Offline Computation Time in Seconds")
    for model, value in results["time_offline"].items():
        print(f"{model}\t{value}")

    print("-" * 20)
    print()
    print("Online Computation Time in Seconds/Iter")
    for model, value in results["time_online"].items():
        print(f"{model}\t{value}")

    print("-" * 20)
    print()
    print("Test error as a value between 0 and 1")
    for model, value in results["test_error"].items():
        print(f"{model}\t{value}")

    return None


class TestCaseTriplets(unittest.TestCase):
    """
    Test cases for Stanford triplets
    evaluation task, cf Werner (2019).
    """

    # names of models for reporting
    MODELS = ["fasttext"]
    # absolute path to corresponding vectors
    VECTOR_PATHS = ["crawl-300d-2M.vec"]
    # number of samples to use from wikipedia corpus
    N_SAMPLES = 300

    def test_wikipedia(self):
        """
        Evaluates all metrics
        against the wikipedia triplets,
        allowing one to reproduce the results
        found in the original paper.
        """

        for m, fp in zip(self.MODELS, self.VECTOR_PATHS):
            results = evaluate_triplets_task(m, fp, 300, 512, self.N_SAMPLES)
            print_results(results)
