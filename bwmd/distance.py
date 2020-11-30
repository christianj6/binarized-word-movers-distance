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
import os
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
from pyemd import emd
from gensim.corpora.dictionary import Dictionary


def convert_vectors_to_dict(
    vectors:list,
    words:list,
)->dict:
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


def build_partitions_lookup_tables(
    vectors:dict,
    I:int,
    real_value_path:str,
    save=True,
    vector_size=300
)->dict:
    '''
    Build a set of lookup tables by recursively
    partitioning the embeddings space and then
    overlaying these partitionings to
    find the approximate-nearest-neighbors
    of each token.

    Parameters
    ---------
        vectors : dict
            Mapping of words to vectors.
        I : int
            Maximum number of iterations cf. Werner (2019)
        real_value_path : str
            Path to original vectors, needed to
            compute cosine distances and locate the
            directory where to output the partitions
            and tables.
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
                Number of expected clusters. The
                final number of clusters will
                likely be less as the algorithm converges
                when partitions can no longer be split. This
                is so avoid sampling errors and keep the
                partitions of relatively equal size.

        Returns
        ---------
            output : dict
                Mapping of centroid token to
                list of associated tokens.
            token_to_centroid : dict
                Reverse of output. Each token
                is associated with its governing
                centroid.
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
        # Second condition determines if last two items are the same,
        # ie if a moderate 'convergence' is seen.
        while len(partitions) < k and not length_of_partitions[-2:].count(length_of_partitions[-2]) == 2:
            # Empty list for updated partitions at end of iteration.
            updated_partitions = []
            # Empty list storing the centroids as well, that these can
            # be zipped with the partition tokens at the end.
            iteration_centroids = []
            # Iterate through the partitions and bisect.
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
                        # Compute distance.
                        # We do not attempt caching because with the randomized
                        # centroids we will most likely store unused values, thereby
                        # increasing memory for little gain. Since the hamming
                        # distance is so cheap anyhow, this is okay.
                        distance = hamdist(vectors[id_to_token[idx]], vectors[id_to_token[centroid]])
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


    n_partitioning_iterations = 100
    # Convert I to k value.
    k = 2**I
    print('Making 100 partitionings of size', str(k))
    # Make directory to store the partitions on disk.
    outputs_dir = f"{''.join(real_value_path.split('.')[0:-1])}"
    partitions_dir = f"{outputs_dir}\\tables\\partitions"
    os.makedirs(partitions_dir, exist_ok=True)
    start = time.time()
    # Perform partitioning on the data.
    for i in tqdm(range(n_partitioning_iterations)):
        # Dump the iteration results to disk so we can
        # garbage collect some more ram. With datasets of this
        # size, this is necessary to prevent excessive allocations.
        with open(f"{partitions_dir}\\{i}", 'wb') as f:
            dill.dump(get_bisecting_partitions(k), f)

    end = time.time()
    print('Time to compute partitionings: ', str(round(end - start, 3)))
    start = time.time()

    # Replace vectors with just the words
    # to save some memory.
    vectors = list(vectors.keys())
    # Load all the partitionings.
    partitioning_iterations = []
    print('Loading partitionings ...')
    for i in tqdm(range(n_partitioning_iterations)):
        with open(f"{partitions_dir}\\{i}", "rb") as f:
            partitioning_iterations.append(dill.load(f))

    # For each token, consolidate and save
    # all words associated with that token, according
    # to those partitions in which it appears.
    print('Organizing associated words for all tokens ...')
    for token in tqdm(vectors):
        all_words_associated_with_current_token = []
        for centroid_to_tokens, token_to_centroid in partitioning_iterations:
            all_words_associated_with_current_token.extend(centroid_to_tokens[token_to_centroid[token]])

        # Dump all the associated words to file.
        count_tokens = Counter(all_words_associated_with_current_token)
        # Cut by count threshold to reduce memory.
        all_words_associated_with_current_token = list(filter(lambda x: x[1] >= 3, count_tokens.items()))
        try:
            with open(f"{outputs_dir}\\tables\\{token}_wordlist", "wb") as f:
                dill.dump(all_words_associated_with_current_token, f)
        except OSError:
            continue

    # Get the raw vectors. This will be used to organize
    # the associated tokens according to the more-reliable
    # cosine distances.
    print('Loading raw vectors ...')
    raw_vectors = real_value_path
    # Load real-valued vectors.
    vectors, words = load_vectors(raw_vectors,
                                size=len(vectors),
                                expected_dimensions=300,
                                expected_dtype='float32',
                                get_words=True)
    vectors = convert_vectors_to_dict(vectors, words)

    # Load all word data upfront.
    print('Loading wordlists ...')
    most_associated_words_each_token = {word: None for word in words}
    for word in tqdm(words):
        try:
            with open(f"{outputs_dir}\\tables\\{word}_wordlist", "rb") as f:
                most_associated_words_each_token[word] = dill.load(f)
        except Exception:
            # Handle some pickling errors and bad file names.
            continue

    # Iterate through words, loading the associated file,
    # and use the cosine distances to further sort the
    # tokens, extracting the top 20 as ANN.
    print('Computing cosine distances for each token ...')
    token_association_key = {}
    for token, vector in tqdm(vectors.items()):
        # words_most_associated_with_current_token = []
        try:
            # Compute and save the cosine distances for the output tables.
            words = list(map(lambda x: (x[0], distance_scipy.cosine(vectors[token],
                                vectors[x[0]])), most_associated_words_each_token[token]))

        except (KeyError, TypeError):
            continue

        # Retrieve the original token and first 20 tokens.
        words = sorted(words, key=lambda x: x[1])[:21]
        # Organize a lookup table for these distances.
        table = {word: distance for word, distance in words}
        # Save table to file.
        if save:
            try:
                # Try to dump the table to a file. Since the files
                # are so small we can make a lot of them with little cost.
                with open(f"{outputs_dir}\\tables\\{token}_table", 'wb') as f:
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
    with open(f"{outputs_dir}\\tables\\_key", 'wb') as f:
        dill.dump(token_association_key, f)

    end = time.time()
    print('Time to compute lookup tables: ', str(round(end - start, 3)))

    return outputs_dir


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
            self.key = {token: [word.lower() for word,_ in tuples] for token, tuples in key.items()}
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
            try:
                # First try to get the value.
                return self.cache[word_1][word_2]
            except KeyError:
                try:
                    # If unavailable, load the necessary table.
                    self.load(word_1)
                    # Try to return the relevant value.
                    return self.cache[word_1][word_2]
                except KeyError:
                    # If the two points are in different clusters,
                    # return default maximum value.
                    return 1

        def load(self, table:str)->None:
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
                self.cache[table] = {word.lower(): distance for word, distance in dill.load(f).items()}
                # Move it to the end of the cache.
                self.cache.move_to_end(table)

            # Drop the least-used item, ie the first.
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


    def __init__(
        self,
        model_path:str,
        dim:int=None,
        with_syntax:bool=False,
        raw_hamming:bool=False,
        full_cache:bool=True,
        size_vocab:int=20_000
    )->None:
        '''
        Initialize table key and cache.

        Parameters
        ---------
            model_path : str
                Path to model file containing
                compressed vectors and optionally
                a set of computed lookup tables.
            dim : int
                Number of dimensions corresponding to
                directory for tables. If left blank, assume
                that the user intends to use uncompressed
                vectors eg for computing other distance metrics.
            with_syntax : bool
                y/n to use syntax in the distance calculations.
            raw_hamming : bool
                y/n to use raw hamming distances based on binary vectors, rather
                than cosine distances from a precomputed lookup table.
            full_cache : bool
                y/n simply load all tables into a single cache. This will allow
                for faster lookup but may be memory-intensive depending
                on the machine and number of computed tables / their
                individual sizes.
            size_vocab : int
                Size of the vocabulary to load. In many cases, it is unnecessary
                to load the full vocabulary considering vectors are ordered by
                frequency. Using more words will increase accuracy but also
                decreases speed due to slower lookup times.
        '''
        # Set vocab size.
        self.size_vocab = size_vocab
        # Determine if user is computing BWMD or
        # attempting to use real-valued vectors for another
        # distance metric.
        if dim == None:
            # User is submitting real-valued vectors.
            # In this case we interpret the model_path as
            # a path to the vectors directly.
            path_to_vectors = model_path
            # Assume 300 dimensions.
            self.dim = 300
            dtype = 'float32'

        else:
            # Otherwise we interpret model_path
            # as a path to a directory containing
            # compressed vectors and tables.
            path_to_vectors = f"{model_path}\\vectors.txtc"
            self.dim = dim
            dtype = 'bool_'
            if not raw_hamming:
                with open(f"{model_path}\\tables\\_key", "rb") as f:
                    if full_cache:
                        # Load all tables into a single cache.
                        self.cache = self.load_all_lookup_tables(dill.load(f), model_path)


                    # TODO: Adjust filenames and fix LRU cache compatibility.

                    else:
                        # Create intelligent cache from lookup tables.
                        self.cache = self.LRUCache(2000, dill.load(f), model, dim)

        # Load the vectors and the words.
        vectors, words = load_vectors(path_to_vectors,
                                size=self.size_vocab,
                                expected_dimensions=self.dim,
                                expected_dtype=dtype, get_words=True,
                                return_numpy=True)
        # Convert to a dictionary for fast lookups.
        self.vectors = convert_vectors_to_dict(vectors, words)
        # Cast words to set for faster lookup.
        self.words = set(words)
        # Load the lookup table of dependency distances.
        if with_syntax:
            with open('bwmd\\data\\dependency_distances', 'rb') as f:
                self.dependency_distances = dill.load(f)

            self.nlp = spacy.load('en_core_web_sm')


    def load_all_lookup_tables(self, key:dict, model_path:str)->dict:
        '''
        Load all lookup tables as a single
        dictionary for accelerated lookup
        during distance computations.

        Parameters
        ---------
            key : dict
                Mapping of tokens to their assocaited
                words. Keys are used as a base list for
                loading all dicts.
            model_path : str
                Path to model directory
                wherein are contained the tables.

        Returns
        ---------
            lookup_dict : dict
                Dictionary mapping each
                token to a dictionary of
                ANNs and their cosine
                distances.
        '''
        # Reformat the keys for compatibility with the
        # corpus. Lowercase.

        # TODO: Clean the original files.

        key = {token: [word.lower() for word,_ in tuples] for token, tuples in key.items()}
        # Create outer dictionary.
        lookup_dict = {}
        # Load all the tables.
        print('Loading all lookup tables ...')
        for k in tqdm(list(key.keys())[:self.size_vocab]):
            with open(f'{model_path}\\tables\\{k}_table', 'rb') as f:
                # Reformat the words to lowercase.
                updated_dict = {word.lower(): value for (word,_), value in dill.load(f).items()}
                # Make inner dictionary.
                lookup_dict[k.lower()] = updated_dict

        return lookup_dict


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


        def get_distance_unidirectional(pdist:'np.array',
                                    depdist:'np.array'=None,
                                    syntax:bool=False)->float:
            '''
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
            '''
            wmd = 0
            for i, array in enumerate(pdist):
                j = np.argmin(array)
                # Try to get syntax distance, otherwise just use semantic distance.
                if syntax:
                    # Some dummy values for a weighted average.

                    # TODO: Make weighted average variables more exposed.

                    wmd += 0.8 * pdist[i][j] + 0.2 * depdist[i][j]

                else:
                    wmd += pdist[i][j]

            # Divide by length of first text to normalize the score.
            return wmd / pdist.shape[0]

        param = {}
        paramT = {}
        # If syntax to be included, get dependencies
        # before removing stopwords.
        try:
            # Try to access attr to see if an error occurs.
            self.dependency_distances
            # Get the dependencies.
            dep_a = get_dependencies(text_a)
            dep_b = get_dependencies(text_b)

        except AttributeError:
            # If we can't access the distance table,
            # we assume the object was not configured for syntax.
            pass

        # Create a mask for removing stopwords.
        mask_a = [False if a in sw or not a in self.words else True for a in text_a]
        mask_b = [False if b in sw or not b in self.words else True for b in text_b]
        # Update the syntax array to account for
        # the removed stopwords.
        try:
            # Try to access attr to see if an error occurs.
            self.dependency_distances
            # Create a distance matrix for
            # the dependencies.
            depdist = self.get_pairwise_distance_matrix(
                np.array(dep_a)[mask_a],
                np.array(dep_b)[mask_b],
                dist=self.dependency_distance(),
                default=self.default_maximum_value(default=1.0)
            )
            # Messy technique to pass these as
            # optional kwargs.
            param['depdist'] = depdist
            paramT['depdist'] = depdist.T
            param['syntax'], paramT['syntax'] = True, True

        except AttributeError:
            # Make a dummy array so numba can compile it.
            pass

        pdist = self.get_pairwise_distance_matrix(
            np.array(text_a)[mask_a],
            np.array(text_b)[mask_b],
            dist=self.related_word_distance_lookup(),
            default=self.hamming_distance()
        )

        # Get score from both directions and sum to make the
        # metric bidirectional. Summation determined to be most
        # effective approach cf. Hamann (2018).
        bwmd = get_distance_unidirectional(pdist, **param)
        # Use transpose to get distance in other direction.
        bwmd += get_distance_unidirectional(pdist.T, **paramT)

        # Divide by two to normalize the score.
        return bwmd / 2


    def dependency_distance(self):
        '''
        Decorator to return dependency
        distances between dependency tags.
        '''
        def distance(a:str, b:str):
            '''
            Enclosed function for
            the distance lookup.

            Parameters
            ---------
                a : str
                    First dependency.
                b : str
                    Second dependency.

            Returns
            --------
                distance : float
                    Weighted graph distance.
            '''
            try:
                # Try lookup.
                return self.dependency_distances[a][b]
            except KeyError:
                # Otherwise None for error handling.
                return None

        return distance


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
            pairwise_distances : np.array
                Matrix of document pairwise distances.
        '''
        # Iterate over a set of docs and compute similarity.
        matrix = []
        for doc_a in tqdm(corpus):
            distances = []
            for doc_b in corpus:
                distances.append(self.get_distance(doc_a, doc_b))

            matrix.append(distances)

        # TODO: Multiprocessing.
        # TODO: Try to store some stuff in a cache ie the dependencies.

        # Pairwise distance matrix.
        return np.array(matrix)


    def hamming_distance(self):
        '''
        Decorator to return
        hamming distance between
        two word vectors.
        '''
        def distance(a:str, b:str):
            '''
            Enclosed function
            for returing the
            hamming distance.

            Parameters
            ---------
                a : str
                    First token.
                b : str
                    Second token.

            Returns
            --------
                distance : float
                    Hamming distance.
            '''
            return np.count_nonzero(self.vectors[a] \
                        != self.vectors[b]) / self.dim

        return distance


    def default_maximum_value(self, default:float=1.0):
        '''
        Decorator for returning a default
        maximum value.
        '''
        def distance(a:str, b:str):
            '''
            Enclosed function
            for returing the default.

            Parameters
            ---------
                a : str
                    First token.
                b : str
                    Second token.

            Returns
            --------
                default : float
                    Default value.
            '''
            return default

        return distance


    def related_word_distance_lookup(self):
        '''
        Decorator to lookup the distance between related
        words using a precomputed
        lookup table of cosine distances. If the
        value cannot be found, function returns
        None.
        '''
        def distance(a:str, b:str):
            '''
            Enclosed function for
            the distance lookup.

            Parameters
            ---------
                a : str
                    First token.
                b : str
                    Second token.

            Returns
            --------
                distance : float
                    Cosine distance.
            '''
            try:
                # Try lookup.
                return self.cache[a][b]
            except (KeyError, AttributeError):
                # Otherwise None for error handling.
                return None

        return distance


    @staticmethod
    def get_pairwise_distance_matrix(text_a:list, text_b:list,
                            dist:'Function', default:'Function')->'np.array':
        '''
        Compute a pairwise distance matrix over
        a series of tokens, using a custom
        distance function.

        Parameters
        ---------
            text_a : list
                First text.
            text_b : list
                Second text.
            dist : Function
                Function for computing
                the distance between
                two tokens.
            default : Function
                Default function for
                returning a distance
                value when a distance
                cannot be returned.

        Returns
        ---------
            matrix : np.array
                Pairwise distance
                matrix.
        '''
        m, n = len(text_a), len(text_b)
        # Create empty array of appropriate structure.
        pdist = np.zeros((m, n))
        # Iterate over tokens, creating
        # an m*n array.
        for i, a in enumerate(text_a):
            for j, b in enumerate(text_b):
                # Try to get a real-valued distance.
                d = dist(a, b)
                if not d:
                    # If a nan value is returned, use default.
                    d = default(a, b)

                pdist[i, j] = d

        return pdist


    def get_wcd(self, text_a:list, text_b:list):
        '''
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
        '''
        # Get all the embeddings.
        a_emb = [self.vectors[a] for a in text_a if a in self.vectors]
        b_emb = [self.vectors[b] for b in text_b if b in self.vectors]
        # Return distance between mean embeddings.
        return distance_scipy.cosine(
                np.mean(a_emb, axis=0),
                np.mean(b_emb, axis=0)
            )


    def get_wmd(self, text_a:list, text_b:list):
        '''
        Get word-movers distance cf. Kusner (2016).
        This is basically a copy of the gensim
        implementation, citations below:

        .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
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
        '''
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
                if not t1 in text_a_set or not t2 in text_b_set:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = np.sqrt(np.sum((self.vectors[t1] - self.vectors[t2])**2))

        # Compute nBOW representation of documents.
        d1 = nbow(text_a)
        d2 = nbow(text_b)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)


    def get_rwmd(self, text_a:list, text_b:list):
        '''
        Get relaxed word mover's distance
        cf Kusner (2016).

        Parameters
        ---------
            text_a : list
                First text.
            text_b : list
                Second text.

        Returns
        ----------
            distance : float
                Relaxed word mover's distance.
        '''
        # Pairwise distance matrix.
        pdist = self.get_pairwise_distance_matrix(
            text_a,
            text_b,
            # Euclidean distance.
            dist=lambda a, b: np.sqrt(np.sum((self.vectors[a] - self.vectors[b])**2)),
            # Default nan value.
            default=lambda a, b: 1
        )
        # Get distance in both directions to render the
        # distance a true metric. Necessary because we are not
        # optimizing a true flow-matrix problem as with
        # the wmd, but rather transferring all mass
        # to the minimum token.
        rwmd = self.min_distance_unidirectional(pdist)
        # Transpose to get other direction.
        rwmd += self.min_distance_unidirectional(pdist.T)

        return rwmd


    @staticmethod
    def min_distance_unidirectional(pdist:'np.array'):
        '''
        Get minimum distance in one direction.

        Parameters
        ---------
            pdist : np.array
                Pairwise distance matrix.

        Returns
        ---------
            distance : float
                RWMD.
        '''
        d = 0
        for i in pdist:
            d += min(i)

        return d


    def get_relrwmd(self, text_a:list, text_b:list):
        '''
        Get related relaxed word movers distance
        cf Werner 2018.

        Parameters
        ---------
            text_a : list
                First text.
            text_b : list
                Second text.

        Returns
        ----------
            distance : float
                Related relaxed word mover's distance.
        '''
        # Pairwise distance matrix.
        pdist = self.get_pairwise_distance_matrix(
            text_a,
            text_b,
            # Precomputed lookup cf Werner.
            dist=self.related_word_distance_lookup(),
            # Default maximum value cf Werner (cMax-value)
            default=self.default_maximum_value(default=1.0)
        )
        relrwmd = self.min_distance_unidirectional(pdist)
        # Transpose to get other direction.
        relrwmd += self.min_distance_unidirectional(pdist.T)

        return relrwmd
