
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


def build_knn_lookup_tables(vectors, save=True):
    '''
    '''
    def knn():
        '''
        '''
        # TODO: Copy Saurabh's implementation with hamming distance.
        # TODO: Returns dict eg cluster: [word1, word2, ... wordn]
        # TODO: ensure randomness reproducibility
        pass

    def build_lookup_table():
        '''
        '''
        # TODO: Accepts dict entry for cluster, computes
        # pairwise cosine distances using real-valued vectors.
        pass

    # TODO: Load vectors to dict.
    # TODO: Create dict for mapping words to cluster ids.
    # TODO: Perform knn by iterating through dict.values(),
    # updating the cluster id dict.
    # TODO: k = sqrt(n / I); input I and it dynamically determines k.
    # TODO: Iterate through the clusters and make lookup tables.
    # TODO: if save, output to appropriate directory
    # a dill-pickled dictionary for each cluster which stores the
    # distances in pairwise format, eg: word1[word2]: cosine_distance

    pass


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
                        # TODO: Load into the cache and get the distance.
                    else:
                        # TODO: Use a default value.

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
