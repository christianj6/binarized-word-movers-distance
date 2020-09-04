
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
        # TODO: self.tables
        # TODO: self.cache
        pass


    def similarity():
        '''
        '''
        # TODO: Computes the similarity between two documents.
        # TODO: Nested functions to handle the other computations.
        # TODO: Appropriate functions for syntactic dependency info.
        # TODO: Error handling or clever strategy if trying to
        # compute a distance which is not represented within a
        # lookup table.
        pass


    def pairwise():
        '''
        '''
        # TODO: Use self.cache for a cache removal policy.
        # TODO: Multiprocessing.
        pass
