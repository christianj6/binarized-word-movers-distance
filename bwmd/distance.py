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
from collections import OrderedDict
import numpy as np
# Set random seed.
random.seed(42)


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
    def kmeans(k:int)->dict:
        '''
        Bisecting K-means cluster algorithm for binary vectors. Uses
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
        start = time.time()
        # Create a cache of distances to remove repeated calculations.
        cache = {}
        vector_space = list(vectors.items())
        # Store partitions in separate object which is updated.
        partitions = [vector_space]
        print('Clustering with bisecting kmeans ...')
        while len(partitions) < k:
            centroid_words = []
            # Empty list to store new partitioning at each iteration.
            new_partitions = []
            # At each iteration, split each of the partitions.
            for partition in partitions:
                new_partition = []
                centroids = random.sample(range(len(partition)), 2)
                output = {centroid: [] for centroid in centroids}
                # Assign all words to a partition.
                for j, (word, vector) in enumerate(partition):
                    distances = []
                    for centroid in centroids:
                        try:
                            # First try to find distance in cache.
                            distance = cache[word][vector_space[centroid][0]]
                        except KeyError:
                            # Otherwise compute and store in cache.
                            if vector_size == 300:
                                distance = distance_scipy.cosine(vector, partition[centroid][1])
                            else:
                                distance = hamdist(vector, partition[centroid][1])

                            cache[word] = {}
                            cache[word][vector_space[centroid][0]] = distance

                        distances.append(distance)

                    # Closest centroid is the assignment for the token.
                    cluster = centroids[distances.index(min(distances))]
                    output[cluster].append(word)

                # Update the lists.
                both_clusters = [cluster_id for cluster_id, words in list(output.items()).copy()]
                a, b = both_clusters[0], both_clusters[1]
                new_partition.append([(word, vectors[word]) for word in output[a]])
                new_partition.append([(word, vectors[word]) for word in output[b]])
                centroid_words.append(partition[a][0])
                centroid_words.append(partition[b][0])

                # Add the updated partitioning.
                new_partitions += new_partition

            # Update partitions.
            partitions = new_partitions

        # List the centroids as initialization for kmeans clustering.
        centroids = [list(vectors.keys()).index(word) for word in centroid_words]

        # Create a reverse-mapping of tokens to clusters used to access the
        # computed tables via later caching policy.
        token_to_centroid = {token: centroid for partition, centroid in zip(partitions, centroids)
                                for token, vector in partition}
        # Format output.
        output = zip(centroids, [[word for word, vector in partition] for partition in partitions])
        output = dict(output)

        # for centroid, tokens in zip(centroid_words, partitions):
        #     print(centroid.upper())
        #     print([token for token, vector in tokens])
        #     print('\n\n')

        end = time.time()
        print('Time to cluster: ', str(round(end - start, 3)))

        return output, token_to_centroid

    def build_lookup_table(cluster:tuple, real_value_vectors:dict)->dict:
        '''
        Constructs a hamming distance lookup table for a given
        dict entry produced by the kmeans method.

        Parameters
        ---------
            cluster : tuple
                Cluster id, [list of words]
            real_value_vectors : dict
                Dictonary of real-valued vectors for
                computing the cosine distance of paired words.

        Returns
        ---------
            table : dict
                Word-to-word mapping with hamming
                distances.
        '''
        table = {}
        idx, words, = cluster
        for word_1 in words:
            table[word_1] = {}
            for word_2 in words:
                # Get cosine distance with real-value vectors.
                distance = distance_scipy.cosine(real_value_vectors[word_1],
                                    real_value_vectors[word_2])
                table[word_1][word_2] = distance

        return table

    # Determine optimal k-value based
    # on total number of desired clusters.
    k = 2**I
    print('Making partition: ', str(k))
    # Estimate required memory.
    mem = (((len(vectors) / k)**2 * 12) * k) / 1073741274
    print(f'Estimated to require {round(mem, 2)} GB.')
    # Perform k-means clustering on the data.
    ids, token_to_centroid = kmeans(k)

    # Store the reverse mapping for indexing the tables.
    with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\_key", 'wb') as f:
        dill.dump(token_to_centroid, f)

    # Store the output for constructing the tables.
    with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\_ids", 'wb') as f:
        dill.dump(ids, f)

    # Build lookup tables based on ids.
    raw_vectors = f"{path.split('.')[0].split('-')[0]}.txt"
    # Load real-valued vectors.
    real_value_vectors, words = load_vectors(raw_vectors,
                                expected_dimensions=300,
                                expected_dtype='float32', get_words=True)
    real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)
    start = time.time()
    # Compute and store a table for each cluster.
    for cluster in ids.items():
        table = build_lookup_table(cluster, real_value_vectors)
        if save:
            with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\{cluster[0]}", 'wb') as f:
                dill.dump(table, f)

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
                return self.cache[table][word_1][word_2]
            except KeyError:
                try:
                    # If unavailable, load the necessary table.
                    self.load(table)
                    # Try to return the relevant value.
                    return self.cache[table][word_1][word_2]
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
            with open(f"{self.directory}\\{table}", "rb") as f:
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
                self.cache = self.LRUCache(15, dill.load(f), model, dim)

        else:
            # Load the raw binary vectors.
            filepath = f"res\\{model}-{dim}.txtc"
            vectors, words = load_vectors(filepath,
                                expected_dimensions=int(dim),
                                    expected_dtype='bool_', get_words=True)
            self.vectors = convert_vectors_to_dict(vectors, words)

        # TODO: Load the lookup table of dependency distances.
        if with_syntax:
            self.dependency_distances = dict()


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
            # TODO: Parse the sentence, retrieving the dependencies.
            # TODO: Use spacy and convert to Stanford.
            # TODO: Reformat for compatibility with my dict.
            # TODO: Return as integers compatible with dependency table.
            pass

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
                distances = []
                for word_b in b:
                    try:

                        # TODO: Conditional to prevent attempting to load
                        # values from different clusters.

                        # Get value from cache. Cache handles itself.
                        distance = self.cache.get(word_a, word_b)
                    except AttributeError:
                        # Means there is not cache ie we are using raw hamming.
                        distance = hamdist(self.vectors[word_a], self.vectors[word_b])

                    # Divide by dimension to normalize score.
                    distances.append(distance / self.dim)

                distance = min(distances)
                try:

                    # TODO: Try get syntax info, if error assume that object is not
                    # configured for getting syntex information.
                    # TODO: Get a list of syntactic dependencies for a and b, respectively.

                    a_dep, b_dep = get_dependencies(a), get_dependencies(b)
                    dependency_distance = self.dependency_distances[a_dep[i]][b_dep[distances.index(distance)]]
                    wmd += distance * dependency_distance
                except (TypeError, AttributeError):
                    wmd += distance

            # Divide by length of first text to normalize the score.
            return wmd / len(a)

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
            pairwise_distances : list
                Matrix of document pairwise distances.
        '''
        # TODO: Clarify the output format.
        # TODO: Iterate over a set of docs and compute similarity.
        # TODO: Multiprocessing.
        # TODO: Try to store some stuff in a cache ie the dependencies.
        # TODO: Return pairwise matrix.
        pass
