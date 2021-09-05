import unittest
import sys
import csv
import os

from tqdm import tqdm
from scipy.stats import spearmanr

from bwmd.compressor import Compressor
from bwmd.tools import (
    convert_vectors_to_dict,
    load_vectors,
    hamming_similarity,
    cosine_distance,
)


class TestCasesCompressor(unittest.TestCase):
    """
    Test cases for autoencoder compressor.
    """

    REDUCED_DIMENSIONS = 512
    SIZE = 1_000
    COMPRESSION = "bool_"
    MODELS = ["glove"]
    DATASETS = ["simverb3500", "wordsim353"]

    def test_compressor_main_functionality(self):
        """
        Tests that the basic compressor functionality
        works without producing an error.
        """
        for m in self.MODELS:
            path = os.path.join(os.getcwd(), "test", "data", f"{m}.txt")
            # Original dimension.
            dim = 300
            vectors, _ = load_vectors(
                path=path, size=self.SIZE, expected_dimensions=dim
            )
            train, validation = vectors[750:], vectors[750:]
            compressor = Compressor(
                original_dimensions=dim,
                reduced_dimensions=self.REDUCED_DIMENSIONS,
                compression=self.COMPRESSION,
            )
            compressor.fit(train, epochs=5)
            compressor.evaluate(validation)
            compressor.transform(path=path, save=True, n_vectors=self.SIZE)

    def test_compressor_word_similarity(self):
        """
        Test performance of compressed vectors
        on the semantic word similarity task.
        """

        def get_similarity_score_for_dataset(vectors: dict, dataset: str):
            fp = os.path.join(os.getcwd(), "test", "data", f"{dataset}.csv")
            y = []
            pred = []
            with open(fp, newline="") as f:
                reader = csv.reader(f, delimiter=",")
                for row in list(reader)[1:]:
                    word1 = row[0]
                    word2 = row[1]
                    sim = row[2]
                    try:
                        pred.append(hamming_similarity(vectors[word1], vectors[word2]))

                    except KeyError:
                        pred.append(0)

                    y.append(float(sim))

            score, _ = spearmanr(pred, y)

            return round(score, 3)

        def score_vectors_on_similarity_task(model, dataset):
            path = os.path.join(os.getcwd(), "test", "data", model, "vectors.txtc")
            vectors, words = load_vectors(
                path=path,
                size=self.SIZE,
                expected_dimensions=self.REDUCED_DIMENSIONS,
                expected_dtype=self.COMPRESSION,
            )
            word_to_vector = convert_vectors_to_dict(vectors, words)
            score = get_similarity_score_for_dataset(word_to_vector, dataset)
            print(
                f"{model} - {self.REDUCED_DIMENSIONS} - {self.SIZE} - {dataset} - {score}"
            )

        for m in self.MODELS:
            for d in self.DATASETS:
                score_vectors_on_similarity_task(model=m, dataset=d)
