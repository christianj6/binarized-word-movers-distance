import math
import random
from tqdm import tqdm
import dill
from bwmd.compressor import load_vectors
from scipy.spatial import distance


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


def build_kmeans_lookup_tables(vectors, I, path, save=True):
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

    Returns
    ---------
        tables : list
            List of lookup tables. Each table corresponds
            to a single computed cluster.
    '''
    def kmeans(k):
        '''
        K-means cluster algorithm for binary vectors. Uses
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

        ###
        # TODO: Two means tree initialization.
        # TODO: Build annoy embedding space.

        # Create a cache of distances to speed up computations.
        cache = {}
        # Initialize random centroids.
        centroids = random.sample(range(len(vectors)), k)
        # Store all words ids in a dictionar for updating centroids.
        word2id = {}

        ###

        # Iterate through the dictionary values to cluster.
        for i in tqdm(range(I - 1)):

            # TODO: Use annoy ann to limit compared clusters.

            # Output mapping cluster_id:[words] for each iteration.
            output = {centroid: [] for centroid in centroids}
            vector_space = list(vectors.items())
            for j, (word, vector) in enumerate(vector_space):
                if i == 0:
                    # Store word indices for remapping centroids.
                    word2id[word] = j
                distances = []
                for centroid in centroids:
                    try:
                        # Attempt to lookup in cache.
                        distance = cache[word][vector_space[centroid][0]]
                    except KeyError:
                        # Otherwise compute hamming distance.
                        distance = hamming(vector, vector_space[centroid][1])
                        # Update the cache.
                        cache[word] = {}
                        cache[word][vector_space[centroid][0]] = distance

                    distances.append(distance)

                cluster = centroids[distances.index(min(distances))]
                # Update outputs.
                output[cluster].append(word)

            # Determine mean point within each cluster.
            new_output = {}
            for cluster, words in output.items():
                word_distances = []
                for word1 in words:
                    distances = []
                    for word2 in words:
                        try:
                            # Try use cache distance.
                            distance = cache[word1][word2]
                        except KeyError:
                            # Otherwise compute distance.
                            distance = hamming(vectors[word1], vectors[word2])
                            # Update cache.
                            cache[word1][word2] = distance

                        distances.append(distance)

                    word_distances.append(sum(distances) / len(distances))

                mean = words[word_distances.index(min(word_distances))]
                # Update position of centroid
                centroid = word2id[mean]
                # Store new output info.
                new_output[centroid] = words

            output = new_output
            # Update the centroids.
            centroids = list(output.keys())

        for centroid, words in output.items():
            print('Cluster ', str(centroid), ': ', str(len(words)), ' points')

        return output, cache

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
            table[word1] = {}
            for word_2 in words:
                # Get cosine distance with real-value vectors.
                distance = distance.cosine(real_value_vectors[word_1],
                                    real_value_vectors[word_2])
                table[word_1][word2] = distance

        return table

    # Determine optimal k-value based on length of dataset and
    # maximum number of iterations cf. Werner et al. (2019).
    k = round(math.sqrt(len(vectors) / I))
    # Perform k-means clustering on the data.
    ids, cache = kmeans(k)
    # Build lookup tables based on ids.
    # Load real-valued vectors.
    real_value_vectors, words = load_vectors(VECTORS,
                            expected_dimensions=300,
                                expected_dtype='float32', get_words=True)
    real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)
    tables = []
    for cluster in ids.items():
        table = build_lookup_table(cluster, real_value_vectors)
        tables.append(table)
        if save:
            # TODO: Fix file names across the repo.
            with open(f"res\\tables\\{path.split('.')[0]}\\{cluster}", 'wb') as f:
                dill.dump(table, f)

    return tables


class BWMD():
    '''
    '''
    def __init__(self):
        '''
        '''
        # TODO: self.tables --> load from file into a mapping of word to table
        # TODO: self.cache --> load from files into a dict of table_id
        pass


    def similarity(a, b, distance=False):
        '''
        '''
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
