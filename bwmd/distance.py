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
import spacy
from spacy.tokens import Doc
from nltk.corpus import stopwords
sw = stopwords.words("english")
# Set random seed.
# random.seed(42)


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
        for token in list(vectors.keys()):
            cache[token] = {}
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
            # print(len(partitions))
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
                        # print('skipping small')
                        # Add the partition.
                        updated_partitions.append(partition)

                        # TODO: Get centroid from previous iteration
                        iteration_centroids.append(final_centroids[j])


                        # subsample_size = round(math.sqrt(len(partition)))
                        # # Default value to prevent the subsample size from getting too low.
                        # subsample_size = len(partition) if subsample_size < 10 else subsample_size
                        # # print(subsample_size)
                        # subsample = random.sample(range(len(partition)), subsample_size)

                        # for idx in subsample:
                        #     # Store average distances to other points to identify best centroid.
                        #     avg_distances = []
                        #     # Iterate through all points to compare distances.
                        #     for id_1 in subsample:
                        #         # Store pairwise distances.
                        #         distances = []
                        #         for id_2 in subsample:
                        #             try:
                        #                 # Try to get distance from cache.
                        #                 distance = cache[id_to_token[id_1]][id_to_token[id_2]]
                        #             except KeyError:
                        #                 # Otherwise compute hamming distance.
                        #                 distance = hamdist(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                        #                 # distance = distance_scipy.cosine(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                        #                 # Store in the cache.
                        #                 cache[id_to_token[id_1]][id_to_token[id_2]] = distance

                        #             distances.append(distance)

                        #         # Average the distances to other points.
                        #         avg_distance = sum(distances) / len(distances)
                        #         avg_distances.append(avg_distance)

                        #     # The point with the lowest average distance is made the new centroid.
                        #     min_avg_distances_id = subsample[avg_distances.index(min(avg_distances))]
                        #     # Convert this to a global id.
                        #     iteration_centroids.append(global_ids[id_to_token[min_avg_distances_id]])

                        continue




                    # Split into two partitions.
                    # First establish two random centroids.
                    # Attempt to find a pair of centroids at a maximum distance, assuming
                    # that this will be an advantageous bipartition of the space,
                    # given that we require only two partitions.

                    # removing this optimization for speed#######################################

                    # possible_centroids = []
                    # # # distances_of_possible_centroids = []
                    # for i in range(len(partition)):





                    # # for i in range(5):

                    #     initial_centroids = random.sample(range(len(partition)), 2)
                    #     possible_centroids.append(initial_centroids)
                    #     # distances_of_possible_centroids.append(hamdist(vectors[id_to_token[initial_centroids[0]]], vectors[id_to_token[initial_centroids[1]]]))

                    # # print(sorted(distances_of_possible_centroids, reverse=True)[0])


                    # # TODO: Choose one centroid then get second at maximal distance from this point.


                    # initial_centroids = list(sorted(possible_centroids, key=lambda x: hamdist(vectors[id_to_token[x[0]]], vectors[id_to_token[x[1]]]), reverse=True))[0]
                    # # initial_centroids = list(sorted(possible_centroids, key=lambda x: distance_scipy.cosine(vectors[id_to_token[x[0]]], vectors[id_to_token[x[1]]]), reverse=True))[0]

                    # initial_centroids = list(initial_centroids)


                    initial_centroids = random.sample(range(len(partition)), 2)
                    ##########################


                    # Store the resulting partitions as centroids with assigned tokens.
                    output = {centroid: [] for centroid in initial_centroids}


                    ###################################
                    # remove for speed
                    #################################


                    # # Get a random subsample of the full partition space, so as
                    # # to speed up computation.
                    # subsample_size = round(math.sqrt(len(partition)))
                    # # Default value to prevent the subsample size from getting too low.
                    # subsample_size = len(partition) if subsample_size < 10 else subsample_size
                    # # print(subsample_size)
                    # subsample = random.sample(range(len(partition)), subsample_size)





                    # # Store the results of the two means clustering in a list to identify
                    # # the defined convergence point.
                    # # Initialize it with arbitrary value so that the while loop
                    # # works as intended.
                    # differences = [-1]
                    # # Perform two means clustering, iterating until
                    # # the difference between the number of points in each
                    # # cluster reaches a minimum, indicating that the two
                    # # groups are of approximately equal size.
                    # while abs(len(output[initial_centroids[0]]) - len(output[initial_centroids[1]])) > differences[-1]:
                    #     # Append the distance governing the while loop. We must
                    #     # compute it again because we use Python 3.7 and do not have
                    #     # access to assignment experssions cf. PEP 572.
                    #     differences.append(abs(len(output[initial_centroids[0]]) - len(output[initial_centroids[1]])))

                    #     # print(differences)

                    #     # Iterate through values in subsample and assign to centroid.
                    #     for subsample_id in subsample:
                    #         distances = []
                    #         for centroid in initial_centroids:
                    #             try:
                    #                 # Try to get distance from cache.
                    #                 distance = cache[id_to_token[subsample_id]][id_to_token[centroid]]
                    #             except KeyError:
                    #                 # Otherwise compute hamming distance.
                    #                 distance = hamdist(vectors[id_to_token[subsample_id]], vectors[id_to_token[centroid]])

                    #                 # distance = distance_scipy.cosine(vectors[id_to_token[subsample_id]], vectors[id_to_token[centroid]])

                    #                 # Store in the cache.
                    #                 cache[id_to_token[subsample_id]][id_to_token[centroid]] = distance

                    #             distances.append(distance)

                    #         # Minimum value indicates the nearest centroid.
                    #         assignment = initial_centroids[distances.index(min(distances))]
                    #         # Assign the point to the appropriate centroid.
                    #         output[assignment].append(subsample_id)

                    #     # New outputs dict to store the updated centroids.
                    #     updated_centroids = []
                    #     updated_output = {}
                    #     # Identify the mean point in the assigned clusters.
                    #     for _, assigned_subsample_ids in output.items():
                    #         # Store average distances to other points to identify best centroid.
                    #         avg_distances = []
                    #         # Iterate through all points to compare distances.
                    #         for id_1 in assigned_subsample_ids:
                    #             # Store pairwise distances.
                    #             distances = []
                    #             for id_2 in assigned_subsample_ids:
                    #                 try:
                    #                     # Try to get distance from cache.
                    #                     distance = cache[id_to_token[id_1]][id_to_token[id_2]]
                    #                 except KeyError:
                    #                     # Otherwise compute hamming distance.
                    #                     distance = hamdist(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                    #                     # distance = distance_scipy.cosine(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                    #                     # Store in the cache.
                    #                     cache[id_to_token[id_1]][id_to_token[id_2]] = distance

                    #                 distances.append(distance)

                    #             # Average the distances to other points.
                    #             avg_distance = sum(distances) / len(distances)
                    #             avg_distances.append(avg_distance)
                    #             # print(avg_distances)

                    #         # The point with the lowest average distance is made the new centroid.
                    #         min_avg_distances_id = assigned_subsample_ids[avg_distances.index(min(avg_distances))]
                    #         # Establish this point as the new centroid, updating the output dict.
                    #         updated_output[min_avg_distances_id] = []
                    #         updated_centroids.append(min_avg_distances_id)

                    #     # Recluster the subsample by the adjusted centroids.
                    #     for subsample_id in subsample:
                    #         distances = []
                    #         for centroid in updated_centroids:
                    #             try:
                    #                 # Try to get distance from cache.
                    #                 distance = cache[id_to_token[subsample_id]][id_to_token[centroid]]
                    #             except KeyError:
                    #                 # Otherwise compute hamming distance.
                    #                 distance = hamdist(vectors[id_to_token[subsample_id]], vectors[id_to_token[centroid]])

                    #                 # distance = distance_scipy.cosine(vectors[id_to_token[subsample_id]], vectors[id_to_token[centroid]])

                    #                 # Store in the cache.
                    #                 cache[id_to_token[subsample_id]][id_to_token[centroid]] = distance

                    #             distances.append(distance)

                    #         # print(cache)
                    #         # print(distances)

                    #         # Minimum value indicates the nearest centroid.
                    #         assignment = updated_centroids[distances.index(min(distances))]
                    #         # Assign the point to the appropriate centroid.
                    #         updated_output[assignment].append(subsample_id)

                    #     # Update the output and centroid list to reflect the new grouping.
                    #     output = updated_output
                    #     initial_centroids = updated_centroids
                    #     # At the beginning of the next loop, the new grouping will be compared with that
                    #     # from the previous iteration, if this update is improved according to the
                    #     # equalizing convergence principal, the loop will terminate.
                    #     # print(differences)

                # ###################################
                # # remove for speed
                # #################################


                except ValueError:
                    # If the cluster is too small to split, ignore it and store its data.
                    # print('we skipping error')
                    # Add the partition.
                    updated_partitions.append(partition)
                    # TODO: Add an approximate centroid.

                    # TODO: Get centroid from previous iteration
                    iteration_centroids.append(final_centroids[j])


                    # subsample_size = round(math.sqrt(len(partition)))
                    # # Default value to prevent the subsample size from getting too low.
                    # subsample_size = len(partition) if subsample_size < 10 else subsample_size
                    # # print(subsample_size)
                    # subsample = random.sample(range(len(partition)), subsample_size)

                    # for idx in subsample:
                    #     # Store average distances to other points to identify best centroid.
                    #     avg_distances = []
                    #     # Iterate through all points to compare distances.
                    #     for id_1 in subsample:
                    #         # Store pairwise distances.
                    #         distances = []
                    #         for id_2 in subsample:
                    #             try:
                    #                 # Try to get distance from cache.
                    #                 distance = cache[id_to_token[id_1]][id_to_token[id_2]]
                    #             except KeyError:
                    #                 # Otherwise compute hamming distance.
                    #                 distance = hamdist(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                    #                 # distance = distance_scipy.cosine(vectors[id_to_token[id_1]], vectors[id_to_token[id_2]])

                    #                 # Store in the cache.
                    #                 cache[id_to_token[id_1]][id_to_token[id_2]] = distance

                    #             distances.append(distance)

                    #         # Average the distances to other points.
                    #         avg_distance = sum(distances) / len(distances)
                    #         avg_distances.append(avg_distance)
                    #         # print(avg_distances)

                    #     # The point with the lowest average distance is made the new centroid.
                    #     min_avg_distances_id = subsample[avg_distances.index(min(avg_distances))]
                    #     # Convert this to a global id.
                    #     iteration_centroids.append(global_ids[id_to_token[min_avg_distances_id]])

                    continue





                # print(differences)
                # Reassign variable for readability.
                centroids = initial_centroids
                # Create final output for the iteration, grouping all tokens in current partition.
                iteration_output = {centroid: [] for centroid in centroids}
                # Also create a reverse mapping of token to centroid, used in later lookups.
                token_to_centroid = {}
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

                            # distance = distance_scipy.cosine(vectors[id_to_token[idx]], vectors[id_to_token[centroid]])

                            # Store value in cache.
                            cache[id_to_token[idx]][id_to_token[centroid]] = distance

                        distances.append(distance)

                    # print(distances)
                    # exit()

                    # Minimum value determines assignment.
                    assignment = centroids[distances.index(min(distances))]
                    # Update the outputs and token lookup dict.
                    iteration_output[assignment].append(idx)
                    token_to_centroid[id_to_token[idx]] = assignment


                # TODO: Assertions to ensure everything is the same size.


                # Parse outputs into two new partitions which are added to an
                # updated partitioning for future bisections.
                # print(len([[(id_to_token[idx], vectors[id_to_token[idx]]) for idx in indices] for centroid, indices in iteration_output.items()]))
                updated_partitions.extend([[(id_to_token[idx], vectors[id_to_token[idx]]) for idx in indices] for centroid, indices in iteration_output.items()])
                # Add the centroids to the list used to construct the final output.
                iteration_centroids.extend([global_ids[id_to_token[centroid]] for centroid in list(iteration_output.keys())])

            # Update the partitions and centroids.
            partitions = updated_partitions
            length_of_partitions.append(len(partitions))
            final_centroids = iteration_centroids

        # Loading raw vectors.
        # Build lookup tables based on ids.
        # raw_vectors = f"{path.split('.')[0].split('-')[0]}.txt"
        # # Load real-valued vectors.
        # real_value_vectors, words = load_vectors(raw_vectors,

        #                             size=2000,

        #                             expected_dimensions=300,
        #                             expected_dtype='float32', get_words=True)
        # real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)

        # id_to_vector = {idx:entry[1] for idx,entry in enumerate(real_value_vectors.items())}

        # print('Assigning real-valued vectors to centroids ... ')
        # print('\n\n')
        # assignment_of_real_value_vectors = {}
        # for centroid in final_centroids:
        #     assignment_of_real_value_vectors[centroid] = []
        # for token, vector in real_value_vectors.items():
        #         distances = []
        #         for centroid in final_centroids:
        #             distance = distance_scipy.cosine(id_to_vector[centroid], vector)
        #             distances.append(distance)

        #         assignment_of_current_token = final_centroids[distances.index(min(distances))]
        #         assignment_of_real_value_vectors[centroid].append(token)

        # for centroid, tokens in assignment_of_real_value_vectors.items():
        #     print(tokens)
        #     print('\n\n')

        # print('\n\n')
        id_to_code = {idx:entry[1] for idx,entry in enumerate(vectors.items())}
        for centroid, tokens in zip(final_centroids, partitions):
            tokens = sorted(tokens, key=lambda x: hamdist(x[1], id_to_code[centroid]))
            # print(len([token for token, vector in tokens]))
            if 'car' in [token for token,vector in tokens]:
                return [token for token, vector in tokens]
            # print('\n\n')

        # When loop terminates, construct the output from the partitions.
        output = dict(zip(final_centroids, [[token for token, code in partition] for partition in partitions]))

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

    # Convert I to k value.
    k = 2**I
    print('Making partition: ', str(k))
    # Perform k-means clustering on the data.
    sets_of_groupings = []
    for i in tqdm(range(100)):
        # ids, token_to_centroid = kmeans(k)
        sets_of_groupings.append(kmeans(k))





    raw_vectors = f"{path.split('.')[0].split('-')[0]}.txt"
    # Load real-valued vectors.
    real_value_vectors, words = load_vectors(raw_vectors,

                                size=200000,

                                expected_dimensions=300,
                                expected_dtype='float32', get_words=True)
    real_value_vectors = convert_vectors_to_dict(real_value_vectors, words)

    id_to_vector = {idx:entry[1] for idx,entry in enumerate(real_value_vectors.items())}








    potential_tokens = list(set([token for s in sets_of_groupings for token in s]))
    token_scores = []
    for token in potential_tokens:
        n = 0
        for set_groups in sets_of_groupings:
            if token in set_groups:
                n+=1
        token_scores.append(n)

    scored_tokens = zip(potential_tokens, token_scores)
    print([tok for tok in sorted(scored_tokens, key=lambda x: distance_scipy.cosine(real_value_vectors['apple'], real_value_vectors[x[0]])) if tok[1] > 2][1:20])

    print('\n\n')

    exit()

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
                        pass
                    try:
                        # Conditional to prevent attempting to load
                        # values from different clusters.
                        if self.cache.key[word_a] == self.cache.key[word_b]:
                            # Get value from cache. Cache handles itself.
                            distance = self.cache.get(word_a, word_b)
                        else:
                            # Otherwise use default maximum value.
                            distance = 1
                    except AttributeError:
                        # Means there is not cache ie we are using raw hamming.
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
