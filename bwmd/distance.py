'''
OVERVIEW
This module contains classes and methods for
computing the so-called Binarized Word
Mover's Distance and precomputing cluster-based
lookup tables of pairwise word vector distances.
Clusters are built using bisecting kmeans
clustering and the distance metric itself is
computed via the BWMD object, which can be
configured to compute the distance with raw hamming
space distances, or via the lookup tables, whereby an
LRU caching strategy is employed with a helper class. For
additional information on clustering parameters etc, please
consult the appropriate classes.

USAGE
Load a set of vectors using the load_vectors method from
bwmd.compressor, then convert these vectors to a dict
using the method in this module. You must then pass these
objects into the method build_kmeans_lookup_table, which will
construct a hamming space partition and then compute and save
a set of lookup tables corresponding to the individual partitions.
By subsequently instantiating a BWMD class referring to the set
of vectors used to construct the tables, you may compute the
BWMD distance using these cached tables.
'''
import math
import random
from tqdm import tqdm
import dill
from bwmd.compressor import load_vectors
from scipy.spatial import distance as distance_scipy
from gmpy2 import hamdist, to_binary
import time
from collections import OrderedDict, Counter
import numpy as np
import spacy
from spacy.tokens import Doc
from nltk.corpus import stopwords
sw = stopwords.words("english")


def convert_vectors_to_dict(vectors:list, words:list)->dict:
    '''
    Convert a set of loaded word vectors
    to a word-vector dictionary for
    easier access.

    Parameters
    ---------
        vectors : list
            List of vectors as np.arrays.
        words : list
            List of words as strings.

    Returns
    ---------
        vectors_dict : dict
            Mapping of words to vectors.
    '''
    vectors_dict = {}
    for vector, word in zip(vectors, words):
        vectors_dict[word] = vector

    return vectors_dict


def build_kmeans_lookup_tables(vectors:dict, I:int, path:str,
                        save=True, vector_size=300)->dict:
    '''
    Build a set of lookup tables by first clustering
    all embeddings and then computing pairwise intra-cluster
    distances. The computed tables can then be saved for
    use in the BWMD distance calculations.

    Parameters
    ---------
        vectors : dict
            Mapping of words to vectors.
        I : int
            Maximum number of iterations cf. Werner (2019)
        save : bool
            If to save computed tables to file.
        vector_size : int
            Dimensions of each vector.

    Returns
    ---------
        token_to_centroid : dict
            Dictonary mapping tokens to their tables. Used for later
            access of table distances..
    '''
    def get_bisecting_partitions(k:int)->dict:
        '''
        Bisecting partioning algorithm for binary vectors. Uses
        hamming distance instead of Euclidean distance.

        Parameters
        ---------
            k : int
                Number of expected clusters.

        Returns
        ---------
            ids : dict
                Mapping of determined cluster to words. Words
                can then be mapped back to their embeddings
                using the input vector dict.
            cache : dict
                Precomputed values.
        '''
        # List of partitions which will be iteratively bisected.
        # Begin with the full vector space.
        partitions = [list(vectors.items())]
        # Make a list of token to global ids.
        global_ids = {entry[0]:idx for idx,entry in enumerate(vectors.items())}
        # List of centroids.
        final_centroids = []
        # Keep track of partition sizes to identify plateau as secondary condition
        # to terminate the while loop.
        # Dummy values so we don't start with an empty list.
        length_of_partitions = [1, 3]
        # Iterate until we reach the desired number of partitions.
        # Second condition determines if last four items are the same,
        # ie if a moderate 'convergence' is seen.
        while len(partitions) < k and not length_of_partitions[-2:].count(length_of_partitions[-2]) == 2:
            # Empty list for updated partitions at end of iteration.
            updated_partitions = []
            # Empty list storing the centroids as well, that these can
            # be zipped with the partition tokens at the end.
            iteration_centroids = []

            for j, partition in enumerate(partitions):
                # Map partition indices to tokens for retrieval.
                id_to_token = {idx:entry[0] for idx,entry in enumerate(partition)}
                # If the partition is already below a certain size, namely the
                # length of the total vector space divided by the intended k,
                # this partition is immediately added to the final set of partitions
                # without subdividing it further. This method aims to
                # achieve a standard size for the final groups, while also
                # avoiding cases where a group which is divided unequally, produces
                # a cluster so small that it cannot be subsampled effectively.
                try:
                    if len(partition) <= len(vectors) / k:
                        # Add the partition.
                        updated_partitions.append(partition)
                        # Get centroid from previous iteration
                        iteration_centroids.append(final_centroids[j])
                        continue

                    # Split into two partitions.
                    # First establish two random centroids.
                    initial_centroids = random.sample(range(len(partition)), 2)
                    # Store the resulting partitions as centroids with assigned tokens.
                    output = {centroid: [] for centroid in initial_centroids}

                except ValueError:
                    # If the cluster is too small to split, ignore it and store its data.
                    # Add the partition.
                    updated_partitions.append(partition)
                    # Get centroid from previous iteration
                    iteration_centroids.append(final_centroids[j])
                    continue

                # Reassign variable for readability.
                centroids = initial_centroids
                # Create final output for the iteration, grouping all tokens in current partition.
                iteration_output = {centroid: [] for centroid in centroids}

                # Assign all points from the original partition to
                # either of the two identified centroids.
                for idx in range(len(partition)):
                    # Use the same minimum-distance methodology as before.
                    distances = []
                    for centroid in centroids:
                        try:
                            # Try to find value in cache.
                            distance = cache[id_to_token[idx]][id_to_token[centroid]]
                        except KeyError:
                            # Otherwise compute distance.
                            distance = hamdist(vectors[id_to_token[idx]], vectors[id_to_token[centroid]])
                            # Store value in cache.
                            cache[id_to_token[idx]][id_to_token[centroid]] = distance
                            # Store reverse value too.
                            cache[id_to_token[centroid]][id_to_token[idx]] = distance

                        distances.append(distance)

                    # Minimum value determines assignment.
                    assignment = centroids[distances.index(min(distances))]
                    # Update the outputs and token lookup dict.
                    iteration_output[assignment].append(idx)
                    # token_to_centroid[id_to_token[idx]] = assignment

                # Parse outputs into two new partitions which are added to an
                # updated partitioning for future bisections.
                updated_partitions.extend([[(id_to_token[idx], vectors[id_to_token[idx]]) for idx in indices] for centroid, indices in iteration_output.items()])
                # Add the centroids to the list used to construct the final output.
                iteration_centroids.extend([global_ids[id_to_token[centroid]] for centroid in list(iteration_output.keys())])

            # Update the partitions and centroids.
            partitions = updated_partitions
            length_of_partitions.append(len(partitions))
            final_centroids = iteration_centroids

        id_to_token = {idx: entry[0] for idx,entry in enumerate(vectors.items())}
        # Update the list and dict to use just the tokens because it's easier for later retrieval.
        final_centroids = [id_to_token[idx] for idx in final_centroids]
        # When loop terminates, construct the output from the partitions.
        output = dict(zip(final_centroids, [[token for token, code in partition] for partition in partitions]))
        # Token to centroid mapping
        token_to_centroid = {token: centroid for centroid, tokens in output.items() for token in tokens}

        return output, token_to_centroid

    # Convert I to k value.
    k = 2**I
    # Create a cache of distances to remove repeated calculations.
    cache = {}
    for token in list(vectors.keys()):
        cache[token] = {}

    print('Making 100 partitionings of size', str(k))
    start = time.time()
    # Perform partitioning on the data.
    iteration_results = []
    for i in tqdm(range(100)):
        # Store the outputs and token mappings for each iteration.
        iteration_results.append(get_bisecting_partitions(k))

    end = time.time()
    print('Time to compute partitionings: ', str(round(end - start, 3)))
    start = time.time()

    # Get the raw vectors. This will be used to organize
    # the associated tokens according to the more-reliable
    # cosine distances.
    raw_vectors = f"{path.split('.')[0].split('-')[0]}.txt"
    # Load real-valued vectors.
    real_value_vectors, words = load_vectors(raw_vectors,
                                expected_dimensions=300,
                                expected_dtype='float32', get_words=True)
    real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)

    # Create a key used to govern the cache policy.
    token_association_key = {}
    for token, vector in tqdm(real_value_vectors.items()):
        # Store all words associated with the current token.
        all_words_associated_with_current_token = []
        for centroid_to_tokens, token_to_centroid in iteration_results:
            # Use the two partitioning dictionaries to access only
            # those clusters affiliated with the current token.
            for associated_word in centroid_to_tokens[token_to_centroid[token]]:
                all_words_associated_with_current_token.append(associated_word)

        words_most_associated_with_current_token = []
        # Reduce the total size of the words to save on the cosine
        # distance computations. The heuristic is that if a token
        # occurs more than 3 times, it may be related.
        for word, count in Counter(all_words_associated_with_current_token).items():
            if count > 3:
                words_most_associated_with_current_token.append(word)

        # Compute and save the cosine distances for the output tables.
        words = list(map(lambda x: (x, distance_scipy.cosine(real_value_vectors[token],
                            real_value_vectors[x])), words_most_associated_with_current_token))
        # Retrieve the original token and first 20 tokens.
        words = sorted(words, key=lambda x: x[1])[:21]
        # Organize a lookup table for these distances.
        table = {word: distance for word, distance in words}
        if save:
            try:
                # Try to dump the table to a file. Since the files
                # are so small we can make a lot of them with little cost.
                with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\{token}_table", 'wb') as f:
                    dill.dump(table, f)

                # If successfully dumped, add an entry to the key
                # governing the cache policy.
                token_association_key[token] = [word for word, _ in words]
            except OSError:
                # Some file names (mostly punctuation) are invalid file names. We could
                # map all the tokens to an integer, but since punctuation are
                # irrelevant to the final distance computation anyhow, it is okay to
                # just skip these files and rather have a neater token-wise retrieval.
                continue

    # Save the key mapping all tokens to associated words.
    with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\_key", 'wb') as f:
        dill.dump(token_association_key, f)

    end = time.time()
    print('Time to compute lookup tables: ', str(round(end - start, 3)))

    return token_to_centroid


class BWMD():
    '''
    Object for managing in-memory cache of precomputed
    word vector distance tables, and for calculating
    the binarized word mover's distance between
    pairs of texts.
    '''
    class LRUCache():
        '''
        Ordered dictionary object with some custom methods
        for managing a cache of cluster distance lookup tables.
        '''
        def __init__(self, capacity:int, key:dict, model:str, dim:str)->None:
            '''
            Initialize the cache object.

            Parameters
            ---------
                capacity : int
                    Maximum number of tables that can be
                    stored in the cache, otherwise the least-recently
                    used item is removed. This capacity should be
                    within your devices working memory limits.
                key: dict
                    Mapping of words to tables.
                model : str
                    Name of model.
                dim : str
                    Dimension of data.
            '''
            self.cache = OrderedDict()
            self.capacity = capacity
            self.key = key
            self.directory = f"res\\tables\\{model}\\{dim}"

        def get(self, word_1:str, word_2:str)->float:
            '''
            Get an item from the cache.

            Parameters
            ---------
                word_1 : str
                    First word for getting distance.
                word_2 : str
                    Second word for getting distance.

            Returns
            ---------
                distance : float
                    Precomputed distance value. Returns
                    an arbitrary, maximum default if not distance can
                    be found.
            '''
            table = self.key[word_1]
            try:
                # First try to get the value.
                return self.cache[table][word_2]
            except KeyError:
                try:
                    # If unavailable, load the necessary table.
                    self.load(table)
                    # Try to return the relevant value.
                    return self.cache[table][word_2]
                except KeyError:
                    # If the two points are in different clusters,
                    # return default maximum value.
                    return 1

        def load(self, table:dict)->None:
            '''
            Load a new table into the cache if it
            is not yet in the cache.

            Parameters
            ---------
                table : str
                    Name of table.
            '''
            # Load the needed table into the cache.
            with open(f"{self.directory}\\{table}_table", "rb") as f:
                self.cache[table] = dill.load(f)
                # Move it to the end of the cache.
                self.cache.move_to_end(table)

            # Drop the least-used item, ie the first.
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


    def __init__(self, model:str, dim:str,
                    with_syntax:bool=True, raw_hamming:bool=False)->None:
        '''
        Initialize table key and cache.

        Parameters
        ---------
            model : str
                Name of model corresponding to a
                parent directory for tables.
            dim : str
                Number of dimensions corresponding to
                directory for tables.
            with_syntax : bool
                y/n to use syntax in the distance calculations.
            raw_hamming : bool
                y/n to use raw hamming distances based on binary vectors, rather
                than cosine distances from a precomputed lookup table.
        '''
        self.dim = int(dim)
        if not raw_hamming:
            with open(f"res\\tables\\{model}\\{dim}\\_key", "rb") as f:
                # Create cache from lookup tables.
                self.cache = self.LRUCache(2000, dill.load(f), model, dim)

        else:
            # Load the raw binary vectors.
            filepath = f"res\\{model}-{dim}.txtc"
            vectors, words = load_vectors(filepath,
                                expected_dimensions=int(dim),
                                    expected_dtype='bool_', get_words=True)
            self.vectors = convert_vectors_to_dict(vectors, words)

        # Load the lookup table of dependency distances.
        if with_syntax:
            with open('res\\tables\\dependency_distances', 'rb') as f:
                self.dependency_distances = dill.load(f)

            self.nlp = spacy.load('en_core_web_sm')


    def get_distance(self, text_a:list, text_b:list)->float:
        '''
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
        '''
        def get_dependencies(text:list)->list:
            '''
            Get a list of syntactic dependencies for a given text with
            dependency integers correponding to the tokens in the text. Used
            to lookup a syntactic dependency diatance from precomputed
            table based on the Stanford dependency hierarchy.

            Parameters
            ---------
                text : list
                    Text as list of string tokens.

            Return
            ---------
                dependencies : list
                    List of dependencies for the text.
            '''
            # Parse the sentence, retrieving the dependencies.
            # The text is cast as a document to preserve the original
            # tokenization and override Spacy's default tokenizer.
            doc = Doc(self.nlp.vocab, words=text)
            # Re-run the pipeline to obtain parsed text for chunking.
            for name, proc in self.nlp.pipeline:
                doc = proc(doc)

            # Get the dependencies as a list.
            dependencies = [tok.dep_.lower() for tok in doc]

            return dependencies


        def get_distance_unidirectional(a:list, b:list)->float:
            '''
            Calculate the BWMD in one direction. Needed to
            bootstrap a bidirectional distance as the metric
            is inherently unidirectional.

            Parameters
            ---------
                a : list
                    One document as list of token strings.
                b : list
                    Another document as list of token strings.

            Returns
            ---------
                distance_uni : float
                    Unidirectional score.
            '''
            wmd = 0
            for i, word_a in enumerate(a):
                if word_a in sw:
                    # Skip stopwords.
                    continue
                distances = []
                for word_b in b:
                    if word_b in sw:
                        # Skip stopwords.
                        continue
                    try:
                        # Check if either of the words is in the other's table; if
                        # so, then we just look it up accordingly.
                        if word_b in self.cache.key[word_a]:
                            distance = self.cache.get(word_a, word_b)
                        elif word_a in self.cache.key[word_b]:
                            distance = self.cache.get(word_b, word_a)
                        else:
                            # Otherwise use hamming distance
                            distance = hamdist(self.vectors[word_a], self.vectors[word_b])

                    except (AttributeError, KeyError):
                        # Means there is no cache ie we are using raw hamming.
                        # Also if the word is not represented in the vocabulary.
                        distance = hamdist(self.vectors[word_a], self.vectors[word_b])

                    # Divide by dimension to normalize score.
                    distances.append(distance / self.dim)

                distance = min(distances)
                try:
                    # Try get syntax info, if error assume that object is not
                    # configured for getting syntex information.
                    a_dep, b_dep = get_dependencies(a), get_dependencies(b)
                    dependency_distance = self.dependency_distances[a_dep[i]][b_dep[distances.index(distance)]]
                    # Weighted arithmetic mean.
                    wmd += 0.75*distance + 0.25*dependency_distance
                except (TypeError, AttributeError):
                    wmd += distance

            # Divide by length of first text to normalize the score.
            # Length is controlled for stopwords.
            return wmd / len([word for word in a if not word in sw])

        # Get score from both directions and sum to make the
        # metric bidirectional. Summation determined to be most
        # effective approach cf. Hamann (2018).
        bwmd = get_distance_unidirectional(text_a, text_b)
        bwmd += get_distance_unidirectional(text_b, text_a)

        # Divide by two to normalize the score.
        return bwmd / 2


    def pairwise(self, corpus:list)->'np.array':
        '''
        Compute pairwise BWMD distances for all documents
        in a given corpus.

        Parameters
        ---------
            corpus : list[list[str]]
                List of documents as lists of token strings.

        Returns
        ---------
            pairwise_distances : list[list]
                Matrix of document pairwise distances.
        '''
        # Iterate over a set of docs and compute similarity.
        matrix = []
        for doc_a in corpus:
            distances = []
            for doc_b in corpus:
                distances.append(self.get_distance(doc_a, doc_b))

            matrix.append(distances)

        # Pairwise distance matrix.
        return matrix

        # TODO: Multiprocessing.
        # TODO: Try to store some stuff in a cache ie the dependencies.
