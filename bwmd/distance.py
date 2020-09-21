import math
import random
from tqdm import tqdm
import dill
from bwmd.compressor import load_vectors
from scipy.spatial import distance as distance_scipy
import annoy


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
        import time
        # from gmpy2 import hamdist
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
                                # distance = distance_scipy.cosine(vector, partition[centroid][1])
                                distance = np.count_nonzero(vector != partition[centroid][1])
                                # distance = hamdist(vector, partition[centroid][1])
                            else:
                                # distance = hamming(vector, partition[centroid][1])
                                distance = distance_scipy.hamming(vector, partition[centroid][1])
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
                new_partitions += new_partition

            # Update partitions.
            partitions = new_partitions

        # List the centroids as initialization for kmeans clustering.
        centroids = [list(vectors.keys()).index(word) for word in centroid_words]
        # Create a reverse mapping of words to centroids for the ANN speedup.
        token_to_centroid = {token: centroid for partition, centroid in zip(partitions, centroids)
                                for token, vector in partition}

        # for centroid, tokens in zip(centroid_words, partitions):
        #     print(centroid.upper())
        #     print([token for token, vector in tokens])
        #     print('\n\n')

        output = zip(centroids, [[word for word, vector in partition] for partition in partitions])

        end = time.time()
        print(end - start)
        exit()


        # Build annoy embedding space.
        print('Building approximate nearest-neighbor space ...')
        # Add all tokens from the dataset.
        index_to_token = {}
        token_to_index = {}
        # Add all tokens from the dataset.
        index_to_token = {}
        token_to_index = {}
        if vector_size == 300:
            # Use cosine distance if real-valued vectors.
            metric = 'angular'
        else:
            # Use hamming distance if binary vectors.
            metric = 'hamming'
        a = annoy.AnnoyIndex(vector_size, metric=metric)
        for i, (token, embedding) in tqdm(enumerate(vectors.items())):
            if vector_size != 300:
                # Map the string values to integers.
                embedding = list(map(lambda x: int(x), list(embedding.bin)))
            else:
                # Deconstruct the np array.
                embedding = embedding[0].tolist()
            a.add_item(i, embedding)
            # Add item and keep track of the mapping.
            index_to_token[i] = token
            token_to_index[token] = i

        # Build embedding space.
        a.build(n_trees=100)

        # TODO: Build dictionary of all nns for faster lookup?
        # TODO: Determine the optimal n value for the ann search.

        # Iterate through the dictionary values to cluster.
        for i in tqdm(range(I - 1)):
            # Output mapping cluster_id:[words] for each iteration.
            output = {centroid: [] for centroid in centroids}
            vector_space = list(vectors.items())
            for j, (word, vector) in tqdm(enumerate(vector_space)):
                # Compare only relevant centroids, per Cheng et al. (2017),
                # ie, only centroids under regime of word nearest-neighbors.
                nn = a.get_nns_by_item(j, n=50)
                # Remove nn which are centroids themselves.
                nn = [i for i in nn if i not in list(output.keys())]
                relevant_centroids = set()
                for word in [index_to_token[item] for item in nn]:
                    try:
                        relevant_centroids.add(token_to_centroid[word])
                    except KeyError:
                        pass

                # relevant_centroids = set([token_to_centroid[word] for word in
                                                        # [index_to_token[item] for item in nn]])
                distances = []

                # print(relevant_centroids)
                # exit()


        #         # for centroid in centroids:
                for centroid in relevant_centroids:




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
            for cluster, words in tqdm(output.items()):
                # print(len(words))
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

                # print(word_distances)
                try:
                    mean = words[word_distances.index(min(word_distances))]
                    # Update position of centroid
                    centroid = token_to_index[mean]
                    # Store new output info.
                    new_output[centroid] = words
                except ValueError:
                    # Skip in cases where a cluster has become empty.
                    new_output[cluster] = words

            output = new_output
            # Update the centroids.
            centroids = list(output.keys())

            # Update token_to_centroid.
            token_to_centroid = {token: centroid for centroid, words in
                                    list(output.items()) for token in words}

        for centroid, words in output.items():
            print('Cluster ', str(centroid), ': ', str(len(words)), ' points')
            print(words)
            print('\n\n\n')

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
                distance = distance_scipy.cosine(real_value_vectors[word_1],
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
