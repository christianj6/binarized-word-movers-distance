import math
import random
from tqdm import tqdm
import dill
from bwmd.compressor import load_vectors
from scipy.spatial import distance as distance_scipy
# import annoy
from gmpy2 import hamdist, to_binary
import time
from collections import OrderedDict
# Set random seed.
random.seed(42)


def hamming(a, b):
    '''
    Hamming distance for bitarray vectors.

    Parameters
    ---------
        a : BitArray
            Vector a.
        b : BitArray.
            Vector b.

    Returns
    ---------
        distance : int
            Hamming distance, ie
            number of overlapping
            segments.
    '''
    return (a^b).count(True)


def convert_vectors_to_dict(vectors, words):
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


def build_kmeans_lookup_tables(vectors, I, path, save=True, vector_size=300):
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
        tables : list
            List of lookup tables. Each table corresponds
            to a single computed cluster.
    '''
    def kmeans(k):
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

                # Adjust clusters to equal size.
                len_cluster_1 = len(list(output.items())[0][1])
                len_cluster_2 = len(list(output.items())[1][1])
                larger_cluster = 0 if len_cluster_1 > len_cluster_2 else 1
                # Update the lists.
                both_clusters = [cluster_id for cluster_id, words in list(output.items()).copy()]
                larger_cluster = both_clusters.pop(larger_cluster)
                smaller_cluster = both_clusters[0]
                sample_size = abs(len_cluster_1 - len_cluster_2)
                sample_size = round(sample_size / 2)
                for word in random.sample(output[larger_cluster], sample_size):
                    # Logically we can just swap assignments because only two clusters.
                    output[larger_cluster].remove(word)
                    output[smaller_cluster].append(word)

                new_partition.append([(word, vectors[word]) for word in output[larger_cluster]])
                new_partition.append([(word, vectors[word]) for word in output[smaller_cluster]])
                centroid_words.append(partition[larger_cluster][0])
                centroid_words.append(partition[smaller_cluster][0])
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

        return output, cache, token_to_centroid

    def build_lookup_table(cluster, real_value_vectors):
        '''
        Constructs a hamming distance lookup table for a given
        dict entry produced by the kmeans method.

        Parameters
        ---------
            cluster : tuple
                Cluster id, [list of words]

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

    # Determine optimal k-value based on length of dataset and
    # maximum number of iterations cf. Werner et al. (2019).
    k = round(math.sqrt(len(vectors) / I))
    # Perform k-means clustering on the data.
    ids, cache, token_to_centroid = kmeans(k)
    # Build lookup tables based on ids.
    raw_vectors = f"{path.split('.')[0].split('-')[0]}.txt"
    # Load real-valued vectors.
    real_value_vectors, words = load_vectors(raw_vectors,
                                expected_dimensions=300,
                                expected_dtype='float32', get_words=True)
    real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)
    tables = []
    start = time.time()
    # Compute and store a table for each cluster.
    for cluster in ids.items():
        table = build_lookup_table(cluster, real_value_vectors)
        tables.append(table)
        if save:
            with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\{cluster[0]}", 'wb') as f:
                dill.dump(table, f)

    # Store the reverse mapping for indexing the tables.
    with open(f"res\\tables\\{path.split('.')[0].split('-')[0][4:]}\\{vector_size}\\_key", 'wb') as f:
        dill.dump(token_to_centroid, f)

    end = time.time()
    print('Time to compute lookup tables: ', str(round(end - start, 3)))

    return tables, token_to_centroid


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
        def __init__(self, capacity, key, model, dim):
            '''
            Initialize the cache object.

            Parameters
            ---------
                capacity : int
                    Maximum number of tables that can be
                    stored in the cache, otherwise the least-recently
                    used item is removed. This capacity should be
                    within your devices working memory limits.
            '''
            self.cache = OrderedDict()
            self.capacity = capacity
            self.key = key
            self.directory = f"res\\tables\\{model}\\{dim}"

        def get(self, word_1, word_2):
            '''
            Get an item from the cache.
            '''
            try:
                # First try to get the value.
                return self.cache[key[word_1]][word_1][word_2]
            except KeyError:
                # If unavailable, load the necessary table.
                load(self.key[key])
                # Then return the relevant value.
                return self.cache[key[word_1]][word_1][word_2]

        def load(self, table):
            '''
            Load a new table into the cache if it
            is not yet in the cache.
            '''
            # Load the needed table into the cache.
            with open(f"{self.directory}\\{table}", "rb") as f:
                self.cache[table] = dill.load(f)
                # Move it to the end of the cache.
                self.cache.move_to_end(table)

            # Drop the least-used item, ie the first.
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


    def __init__(self, model, dim):
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
        '''
        with open(f"res\\tables\\{model}\\{dim}\\_key", "rb") as f:
            self.cache = LRUCache(2, dill.load(f), model, dim)


    def similarity(a, b, distance=False):
        '''
        '''

        # TODO: cache will be a nested dict: look for table, look for word pair, but
            # if error load then lookup

        # TODO: how to implement removal policy????

        # TODO: Preprocess??
        # TODO: Computes the similarity between two documents.
        # TODO: Nested functions to handle the other computations.
        # TODO: Appropriate functions for syntactic dependency info.
        # TODO: Error handling or clever strategy if trying to
        # compute a distance which is not represented within a
        # lookup table.
        # TODO: if distance return 1 - similarity
        # TODO: dict lookup table for dependency distances
        # TODO: Convert words to numbers for faster lookup??

        # TODO: Sum both directions for bidirectional cf Hamann
        # TODO: Normalize the summation components by document distance d?

        wmd = 0
        # TODO: Get a list of dependencies for a and b, respectively.
        a_dep, b_dep = get_dependencies(a), get_dependencies(b)
        dependency_distances = dict()
        for i, word_a in enumerate(a):
            distances = []
            for word_b in b:
                try:
                    distance = self.cache[self.tables[word_a]][word_a][word_b]
                except KeyError:
                    # TODO: Check if in different tables.
                    if self.tables[word_a] == self.tables[word_b]:
                        pass
                        # TODO: Load into the cache and get the distance.
                    else:
                        # TODO: Use a default value.
                        pass

                distances.append(distance)

            distance = min(distances)
            dependency_distance = dependency_distances[a_dep[i]][b_dep[distances.index(distance)]]

            wmd += distance * dependency_distance




        pass


    def pairwise():
        '''
        '''
        # TODO: Use self.cache for a cache removal policy.
        # TODO: Multiprocessing.
        pass
