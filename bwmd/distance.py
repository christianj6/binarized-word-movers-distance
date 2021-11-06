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
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
import scipy.spatial.distance as distance_scipy
from pyemd import emd
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import numpy as np
import abc
from typing import Callable


class AbstractDistanceMetric(abc.ABC):
    def __init__(self, language: str):
        self.stopwords = stopwords.words(language)

    @abc.abstractmethod
    def _get_text_mask(self, text) -> list:
        return None

    def preprocess_text(self, text):
        mask = self._get_text_mask(text)

        return np.array(text)[mask].tolist()

    def preprocess_corpus(self, corpus: list):
        out = []
        for text in corpus:
            out.append(self.preprocess_text(text))

        return out

    @abc.abstractmethod
    def get_distance(self, text_a: list, text_b: list) -> float:
        return None


class WMD(AbstractDistanceMetric):
    """
    Original Word Mover's Distance cf Kusner (2016).
    """

    def __init__(self, vectors: np.ndarray, language: str):
        self.vectors = vectors
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords and words we cannot embed.
        mask = [
            False if a in self.stopwords or a not in self.vectors else True
            for a in text
        ]

        return mask

    def get_distance(self, text_a: list, text_b: list) -> list:
        """
        Get word-movers distance cf. Kusner (2016).
        This is basically a copy of the gensim
        implementation, citations below:

        .. Ofir Pele and Michael Werman,
            "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman,
            "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".

        [Source]
        https://tedboy.github.io/nlps/_modules/gensim/models/word2vec.html#Word2Vec.wmdistance

        Parameters
        ---------
            text_a : list
                First text.
            text_b : list
                Second text.

        Returns
        ----------
            distance : float
                Word mover's distance.
        """
        # Create dictionary necessary for nbow representation.
        dictionary = Dictionary(documents=[text_a, text_b])
        vocab_len = len(dictionary)

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.

            return d

        # Sets for faster look-up.
        text_a_set = set(text_a)
        text_b_set = set(text_b)

        # Compute euclidean distance matrix.
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in text_a_set or t2 not in text_b_set:
                    continue

                # Compute Cosine distance between word vectors.
                distance_matrix[i, j] = distance_scipy.cosine(
                    self.vectors[t1], self.vectors[t2]
                )

        # Compute nBOW representation of documents.
        d1 = nbow(text_a)
        d2 = nbow(text_b)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)


class WCD(AbstractDistanceMetric):
    """
    Word Centroid Distance; Cosine distance between
    unweighted average word embeddings of each text.
    """

    def __init__(self, vectors: dict, language: str):
        self.vectors = vectors
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords and words we cannot embed.
        mask = [
            False if a in self.stopwords or a not in self.vectors else True
            for a in text
        ]

        return mask

    def get_distance(self, text_a: list, text_b: list) -> float:
        """
        Get word-centroid distance
        cf. Kusner (2016).

        Parameters
        ---------
            text_a : list
                First text.
            text_b : list
                Second text.

        Returns
        ----------
            distance : float
                Word centroid distance.
        """
        # Get all the embeddings.
        a_emb = [self.vectors[a] for a in text_a]
        b_emb = [self.vectors[b] for b in text_b]
        # Return distance between mean embeddings.
        return distance_scipy.cosine(
            np.mean(a_emb, axis=0), np.mean(b_emb, axis=0)
        )


class BOW(AbstractDistanceMetric):
    """
    Cosine distance between BOW vectors
    for each text. First a vectorizer must be fitted
    to the entire corpus before each text
    can be vectorized during distance calculation.
    """

    def __init__(self, corpus: list, language: str, l1_norm=False):
        # organize the pipes starting with bow
        pipes = [
            ("bow", CountVectorizer()),
        ]
        # ability to use normalization strategy
        # to explore concerns by Sato (2021).
        if l1_norm is True:
            # add norm pipe if requested
            pipes.append(("norm", Normalizer(norm="l1")))

        self.pipeline = Pipeline(pipes)
        # fit the pipeline
        self.pipeline.fit(corpus)
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords. embedding model is irrelevant.
        mask = [False if a in self.stopwords else True for a in text]

        return mask

    def preprocess_corpus(self, corpus: list):
        out = []
        for text in corpus:
            # join the texts for tfidf vectorization.
            out.append(" ".join(self.preprocess_text(text)))

        return out

    def get_distance(self, text_a: str, text_b: str) -> float:
        # transform raw texts to tfidf representation.
        ret = self.pipeline.transform([text_a, text_b])

        return distance_scipy.cosine(ret[0].A, ret[1].A)


class RWMD(AbstractDistanceMetric):
    """
    Relaxed Word Mover's Distance lower-bound
    cf Kusner (2016). Move all mass from each token
    to its nearest neigbbor in the other document in both
    directions, then take the maximum of these two
    unidirectional distances.
    """

    def __init__(self, vectors: np.ndarray, language: str):
        self.vectors = vectors
        # Create separate attr because it is faster to check the list.
        self.words = list(self.vectors.keys())
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords and words not represented by
        # the embedding model.
        mask = [
            False if a in self.stopwords or a not in self.words else True
            for a in text
        ]

        return mask

    @staticmethod
    def get_pairwise_distance_matrix(
        text_a: list, text_b: list, dist: Callable, default: Callable
    ) -> np.ndarray:
        """
        Computes a pairwise distance matrix given the tokens
        in text_a and text_b, using the dist callable to compute
        distances where possible, otherwise supplying a default
        maximum value defined by default.

        Parameters
        ---------
            text_a : list[str]
                List of tokens in text a as strings.
            text_b : list[str]
                List of tokens in text b as as trings.
            dist : Callable
                Function for computing distance between
                tokens of text a and b.
            default : Callable
                Function for returning default
                value between tokens of text a and b
                in cases where the dist function fails to
                return a value.

        Returns
        ---------
            pdist : np.ndarray
                Pairwise distance matrix of shape m x n, where
                m and n are the lengths of texts a and b, respectively.
        """
        pdist = np.zeros((len(text_a), len(text_b)))
        for i, x in enumerate(text_a):
            for j, y in enumerate(text_b):
                try:
                    d = dist(x, y)

                except (KeyError, AttributeError):
                    # faster to check errors vs handling nan values.
                    d = default(x, y)

                pdist[i, j] = d

        return pdist

    @staticmethod
    def aggregate_unidirectional_distances(d1: float, d2: float) -> float:
        # per the original paper, choose maximum value as
        # the representative distance
        return max(d1, d2)

    @staticmethod
    def min_distance_unidirectional(pdist: "np.array"):
        """
        Get minimum distance in one direction.

        Parameters
        ---------
            pdist : np.array
                Pairwise distance matrix.

        Returns
        ---------
            distance : float
                Minimum transport cost.
        """
        d = 0
        for i in pdist:
            d += min(i)

        return d

    @classmethod
    def _finalize_distance_computation(cls, pdist: np.ndarray) -> float:
        # get unidirectional distances
        d1 = cls.min_distance_unidirectional(pdist)
        # transposing the array allows us to get min dist
        # in the opposite direction
        d2 = cls.min_distance_unidirectional(pdist.T)
        # aggregate unidirectional distances
        return cls.aggregate_unidirectional_distances(d1, d2)

    def get_distance(self, text_a: list, text_b: list) -> float:
        # distance function: euclidean distance
        distance = lambda a, b: distance_scipy.cosine(
            self.vectors[a], self.vectors[b]
        )
        # default is arbitrary value because it will never be used
        default = lambda a, b: 1
        # get pairwise distances
        pdist = self.get_pairwise_distance_matrix(
            text_a, text_b, dist=distance, default=default
        )
        dist = self._finalize_distance_computation(pdist)

        return dist


class RelRWMD(RWMD):
    """
    Relaxed Related Word Mover's Distance cf Werner (2019).
    Use a precomputed cache to try and lookup distances
    for tokens in compared documents, otherwise use a
    default maximum value. Otherwise the lower bound is
    computed as the RWMD, where the max of the two
    unidirectional values is taken.
    """

    def __init__(self, language: str, cache_path: str):
        self.language = language
        # load the cache
        self.cache = self._load_cache(cache_path)
        self.stopwords = stopwords.words(language)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords
        mask = [False if a in self.stopwords else True for a in text]

        return mask

    @staticmethod
    def _load_cache(cache_path) -> dict:
        """
        Loads lookup tables from file.
        """
        with open(f"{cache_path}\\table", "rb") as f:
            cache = pickle.load(f)

        return cache

    def get_distance(self, text_a: list, text_b: list) -> float:
        # distance function: cache lookup
        distance = lambda a, b: self.cache[a][b]
        # default maximum value cf Werner (cMax)
        default = lambda a, b: 0.7
        # get pairwise distances
        pdist = self.get_pairwise_distance_matrix(
            text_a, text_b, dist=distance, default=default
        )
        dist = self._finalize_distance_computation(pdist)

        return dist


class BWMD(RelRWMD):
    """
    Binarized Word Mover's Distance. Similar in
    spirit to the RelRWMD, but instead of a default maximum
    value the normalized hamming distance between binary
    encoded vectors is used. Instead of taking the maximum of
    two unidirectional transport costs, these values are
    summed and then divided by two.
    """

    def __init__(
        self,
        model_path: str,
        size_vocab: int,
        language: str,
        dim: int = None,
    ) -> None:
        # Load the vectors and the words.
        vectors, words = load_vectors(
            f"{model_path}\\vectors.txtc",
            size=size_vocab,
            expected_dimensions=dim,
            expected_dtype="bool_",
        )
        # Convert to a dictionary for fast lookups.
        self.vectors = convert_vectors_to_dict(vectors, words)
        # Cast words to set for faster lookup.
        self.words = set(words)
        # finish initialization
        super().__init__(language, model_path)

    def _get_text_mask(self, text) -> list:
        # ignore stopwords and words not represented by
        # the embedding model.
        mask = [
            False if a in self.stopwords or a not in self.words else True
            for a in text
        ]

        return mask

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

        # distance is cache lookup just like with relrwmd
        distance = lambda a, b: self.cache[a][b]
        # default is hamming distance
        default = lambda a, b: hamming_distance(
            self.vectors[a], self.vectors[b]
        )
        pdist = self.get_pairwise_distance_matrix(
            text_a, text_b, dist=distance, default=default
        )

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
