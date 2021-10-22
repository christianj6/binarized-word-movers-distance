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
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.distance as distance_scipy
from pyemd import emd
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import numpy as np
import abc


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
    def __init__(self, language):
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        mask = [False if a in self.stopwords else True for a in text]

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

                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = np.sqrt(
                    np.sum((self.vectors[t1] - self.vectors[t2]) ** 2)
                )

        # Compute nBOW representation of documents.
        d1 = nbow(text_a)
        d2 = nbow(text_b)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)


class WCD(AbstractDistanceMetric):
    def __init__(self, vectors: dict, language: str):
        self.vectors = vectors
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
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


class TFIDF(AbstractDistanceMetric):
    def __init__(self, corpus: list, language: str, l1_norm=False):
        norm = "l1" if l1_norm is True else "l2"
        self.vectorizer = TfidfVectorizer(norm=norm)
        self.vectorizer.fit(corpus)
        super().__init__(language)

    def _get_text_mask(self, text) -> list:
        mask = [False if a in self.stopwords else True for a in text]

        return mask

    def preprocess_corpus(self, corpus: list):
        out = []
        for text in corpus:
            # join the texts for tfidf vectorization.
            out.append(" ".join(self.preprocess_text(text)))

        return out

    def get_distance(self, text_a: str, text_b: str) -> float:
        a, b = self.vectorizer.transform([text_a, text_b])

        return distance_scipy.cosine(a, b)


class RWMD(AbstractDistanceMetric):
    def __init__(self):
        # todo: init
        return None

    # todo: preprocessing we can probably inherit in child classes

    def get_distance(self, text_a: list, text_b: list) -> float:
        # todo: generalize get_pairwise_distance_matrix
        # todo: get distance unidirectional
        # todo: get distance unidirectional
        # todo: 'aggregate_unidirectional distances'
        return None


class RelRWMD(RWMD):
    def __init__(self):
        # todo: can probably copy from parent
        return None

    # todo: inherit preprocessing from parent
    # todo: modify methods to achieve the relrwmd


class BWMD(RelRWMD):
    """
    .
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
        .
        """
        # Set vocab size.
        self.size_vocab = size_vocab
        path_to_vectors = f"{model_path}\\vectors.txtc"
        self.dim = dim
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

        # todo: refactor given inheritance structure
        super().__init__(language)

    def _get_text_mask(self, text: list) -> list:
        mask = [
            False if a in self.stopwords or not a in self.words else True
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
