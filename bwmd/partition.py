"""
OVERVIEW
This module contains one large method
for computing a lookup table file which is
used as part of the binarized word mover's 
distance lower bound distance metric.

USAGE
You must first build a set of compressed vectors
using the compressor.py module, then pass these and 
the real-valued vectors as keyword arguments to the
function build_partitions_lookup_tables.
"""


import os
import pickle
import time
import random
from bwmd.tools import (
    hamming_distance,
    load_vectors,
    convert_vectors_to_dict,
    cosine_distance,
)
from tqdm import tqdm
from collections import Counter


def build_partitions_lookup_tables(
    vectors: dict, I: int, real_value_path: str, save=True, vector_dim=300
) -> dict:
    """
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
    """

    def get_bisecting_partitions(k: int) -> dict:
        """
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
        """
        # List of partitions which will be iteratively bisected.
        # Begin with the full vector space.
        partitions = [list(vectors.items())]
        # Make a list of token to global ids.
        global_ids = {entry[0]: idx for idx, entry in enumerate(vectors.items())}
        # List of centroids.
        final_centroids = []
        # Keep track of partition sizes to identify plateau as secondary condition
        # to terminate the while loop.
        # Dummy values so we don't start with an empty list.
        length_of_partitions = [1, 3]
        # Iterate until we reach the desired number of partitions.
        # Second condition determines if last two items are the same,
        # ie if a moderate 'convergence' is seen.
        while (
            len(partitions) < k
            and not length_of_partitions[-2:].count(length_of_partitions[-2]) == 2
        ):
            # Empty list for updated partitions at end of iteration.
            updated_partitions = []
            # Empty list storing the centroids as well, that these can
            # be zipped with the partition tokens at the end.
            iteration_centroids = []
            # Iterate through the partitions and bisect.
            for j, partition in enumerate(partitions):
                # Map partition indices to tokens for retrieval.
                id_to_token = {idx: entry[0] for idx, entry in enumerate(partition)}
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
                        distance = hamming_distance(
                            vectors[id_to_token[idx]], vectors[id_to_token[centroid]]
                        )
                        distances.append(distance)

                    # Minimum value determines assignment.
                    assignment = centroids[distances.index(min(distances))]
                    # Update the outputs and token lookup dict.
                    iteration_output[assignment].append(idx)
                    # token_to_centroid[id_to_token[idx]] = assignment

                # Parse outputs into two new partitions which are added to an
                # updated partitioning for future bisections.
                updated_partitions.extend(
                    [
                        [
                            (id_to_token[idx], vectors[id_to_token[idx]])
                            for idx in indices
                        ]
                        for centroid, indices in iteration_output.items()
                    ]
                )
                # Add the centroids to the list used to construct the final output.
                iteration_centroids.extend(
                    [
                        global_ids[id_to_token[centroid]]
                        for centroid in list(iteration_output.keys())
                    ]
                )

            # Update the partitions and centroids.
            partitions = updated_partitions
            length_of_partitions.append(len(partitions))
            final_centroids = iteration_centroids

        id_to_token = {idx: entry[0] for idx, entry in enumerate(vectors.items())}
        # Update the list and dict to use just the tokens because it's easier for later retrieval.
        final_centroids = [id_to_token[idx] for idx in final_centroids]
        # When loop terminates, construct the output from the partitions.
        output = dict(
            zip(
                final_centroids,
                [[token for token, code in partition] for partition in partitions],
            )
        )
        # Token to centroid mapping
        token_to_centroid = {
            token: centroid for centroid, tokens in output.items() for token in tokens
        }

        return output, token_to_centroid

    n_partitioning_iterations = 100
    # Convert I to k value.
    k = 2 ** I
    print("Making 100 partitionings of size", str(k))
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
        with open(f"{partitions_dir}\\{i}", "wb") as f:
            pickle.dump(get_bisecting_partitions(k), f)

    end = time.time()
    print("Time to compute partitionings: ", str(round(end - start, 3)))
    start = time.time()

    # Replace vectors with just the words
    # to save some memory.
    vectors = list(vectors.keys())
    # Load all the partitionings.
    partitioning_iterations = []
    print("Loading partitionings ...")
    for i in tqdm(range(n_partitioning_iterations)):
        with open(f"{partitions_dir}\\{i}", "rb") as f:
            partitioning_iterations.append(pickle.load(f))

    # For each token, consolidate and save
    # all words associated with that token, according
    # to those partitions in which it appears.
    print("Organizing associated words for all tokens ...")
    for token in tqdm(vectors):
        all_words_associated_with_current_token = []
        for centroid_to_tokens, token_to_centroid in partitioning_iterations:
            all_words_associated_with_current_token.extend(
                centroid_to_tokens[token_to_centroid[token]]
            )

        # Dump all the associated words to file.
        count_tokens = Counter(all_words_associated_with_current_token)
        # Cut by count threshold to reduce memory.
        all_words_associated_with_current_token = list(
            filter(lambda x: x[1] >= 3, count_tokens.items())
        )
        try:
            with open(f"{outputs_dir}\\tables\\{token}_wordlist", "wb") as f:
                pickle.dump(all_words_associated_with_current_token, f)

        except OSError:
            continue

    # Get the raw vectors. This will be used to organize
    # the associated tokens according to the more-reliable
    # cosine distances.
    print("Loading raw vectors ...")
    raw_vectors = real_value_path
    # Load real-valued vectors.
    vectors, words = load_vectors(
        raw_vectors,
        size=len(vectors),
        expected_dimensions=300,
        expected_dtype="float32",
    )
    vectors = convert_vectors_to_dict(vectors, words)

    # Load all word data upfront.
    print("Loading wordlists ...")
    most_associated_words_each_token = {word: None for word in words}
    for word in tqdm(words):
        try:
            with open(f"{outputs_dir}\\tables\\{word}_wordlist", "rb") as f:
                most_associated_words_each_token[word] = pickle.load(f)

        except Exception as e:
            # Handle some pickling errors and bad file names.
            continue

    # Iterate through words, loading the associated file,
    # and use the cosine distances to further sort the
    # tokens, extracting the top 20 as ANN.
    print("Computing cosine distances for each token ...")
    out = {}
    for token, vector in tqdm(vectors.items()):
        try:
            # Compute and save the cosine distances for the output tables.
            words = []
            for word, count in most_associated_words_each_token[token]:
                try:
                    dist = cosine_distance(vectors[token], vectors[word])
                    words.append((word, dist))
                except KeyError:
                    pass

        except TypeError:
            continue

        # Retrieve the original token and first 20 tokens.
        words = sorted(words, key=lambda x: x[1])[:21]
        # Organize a lookup table for these distances.
        table = {word: distance for word, distance in words}
        out[token] = table

    # Save the key mapping all tokens to associated words.
    with open(f"{outputs_dir}\\table", "wb") as f:
        pickle.dump(out, f)

    end = time.time()
    print("Time to compute lookup tables: ", str(round(end - start, 3)))

    return outputs_dir
