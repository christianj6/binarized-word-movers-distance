"""
OVERVIEW
This module contains classes and methods for
computing the Binarized Word Mover's Distance
lower bound textual distance metric.

USAGE
Load a set of vectors using the load_vectors method from
bwmd.compressor, then convert these vectors to a dict
using the utility method.By subsequently instantiating a 
BWMD class instance referring to the set
of vectors used to construct the tables, you may compute the
BWMD distance using these cached tables.
"""

from bwmd.tools import load_vectors, convert_vectors_to_dict, hamming_distance
from bwmd_utils import hamdist
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import numpy as np


class BWMD:
    """
    Object for managing in-memory cache of precomputed
    word vector distance tables, and for calculating
    the binarized word mover's distance between
    pairs of texts.
    """

    def __init__(
        self,
        model_path: str,
        size_vocab: int,
        language: str,
        dim: int = None,
        raw_hamming: bool = False,
    ) -> None:
        """
        Initialize table key and cache.

        Parameters
        ---------
            model_path : str
                Path to model file containing
                compressed vectors and optionally
                a set of computed lookup tables.
            dim : int
                Number of dimensions corresponding to
                directory for tables. If left blank, assume
                that the user intends to use uncompressed
                vectors eg for computing other distance metrics.
            with_syntax : bool
                y/n to use syntax in the distance calculations.
            raw_hamming : bool
                y/n to use raw hamming distances based on binary vectors, rather
                than cosine distances from a precomputed lookup table.
            full_cache : bool
                y/n simply load all tables into a single cache. This will allow
                for faster lookup but may be memory-intensive depending
                on the machine and number of computed tables / their
                individual sizes.
            size_vocab : int
                Size of the vocabulary to load. In many cases, it is unnecessary
                to load the full vocabulary considering vectors are ordered by
                frequency. Using more words will increase accuracy but also
                decreases speed due to slower lookup times.
        """
        # Set vocab size.
        self.size_vocab = size_vocab
        path_to_vectors = f"{model_path}\\vectors.txtc"
        self.dim = dim
        self.stopwords = stopwords.words("english")
        dtype = "bool_"

        if not raw_hamming:
            with open(f"{model_path}\\table", "rb") as f:
                self.cache = pickle.load(f)

        # Load the vectors and the words.
        vectors, words = load_vectors(
            path_to_vectors,
            size=self.size_vocab,
            expected_dimensions=self.dim,
            expected_dtype=dtype,
        )

        # Convert to a dictionary for fast lookups.
        self.vectors = convert_vectors_to_dict(vectors, words)
        # Cast words to set for faster lookup.
        self.words = set(words)

    def get_distance(self, text_a: list, text_b: list) -> float:
        """
        Compute the BWMD when provided with two texts.

        Parameters
        ---------
            text_a : list
                Single document as list of token strings.
            text_b : list
                Second document as list of token strings.

        Returns
        ---------
            bwmd : float
                Distance score.
        """

        def get_distance_unidirectional(
            pdist: "np.array",
        ) -> float:
            """
            Calculate the BWMD in one direction. Needed to
            bootstrap a bidirectional distance as the metric
            is inherently unidirectional.

            Parameters
            ---------
                pdist : np.array
                    Pairwise distance matrix.
                depdist : np.array
                    Optional pairwise
                    syntactic dependency distances.

            Returns
            ---------
                distance_uni : float
                    Unidirectional score.
            """
            wmd = 0
            for i, array in enumerate(pdist):
                j = np.argmin(array)
                wmd += pdist[i][j]

            # Divide by length of first text to normalize the score.
            return wmd / pdist.shape[0]

        pdist = self.get_pairwise_distance_matrix(text_a, text_b)

        if pdist.shape[0] == 0 or pdist.T.shape[0] == 0:
            # Return a default maximum value if we couldn't
            # get any distance info, likely because all
            # the tokens were either stopwords or not
            # in the vocabulary.
            return 1

        # Get score from both directions and sum to make the
        # metric bidirectional. Summation determined to be most
        # effective approach cf. Hamann (2018).
        bwmd = get_distance_unidirectional(pdist)
        # Use transpose to get distance in other direction.
        bwmd += get_distance_unidirectional(pdist.T)

        # Divide by two to normalize the score.
        return bwmd / 2

    def get_pairwise_distance_matrix(self, a, b):
        pdist = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                try:
                    d = self.cache[x][y]

                except (KeyError, AttributeError):
                    d = hamming_distance(self.vectors[x], self.vectors[y])

                pdist[i, j] = d

        return pdist

    def preprocess_text(self, text):
        mask = [
            False if a in self.stopwords or not a in self.words else True for a in text
        ]
        return np.array(text)[mask].tolist()

    def preprocess_corpus(self, corpus: list):
        """
        Preprocess all texts by removing stopwords
        and words not found in the vector tables.
        """
        out = []
        for text in corpus:
            out.append(self.preprocess_text(text))

        return out

    def pairwise(self, corpus: list) -> "np.array":
        """
        Compute pairwise BWMD distances for all documents
        in a given corpus.

        Parameters
        ---------
            corpus : list[list[str]]
                List of documents as lists of token strings.

        Returns
        ---------
            pairwise_distances : np.array
                Matrix of document pairwise distances.
        """
        # Iterate over a set of docs and compute similarity.
        matrix = []
        for doc_a in tqdm(corpus):
            distances = []
            for doc_b in corpus:
                distances.append(self.get_distance(doc_a, doc_b))

            matrix.append(distances)

        # TODO: Multiprocessing for large corpora.

        # Pairwise distance matrix.
        return np.array(matrix)
