import numpy as np
from tqdm import tqdm


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a == b)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a != b) / a.shape[0]


def cosine_distance(x, y):
    return 1 - np.inner(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))


def load_vectors(
    path: str,
    size: int,
    expected_dimensions: int = 300,
    expected_dtype: str = "float32",
    skip_first_line: bool = True,
) -> list:
    """
    Load word vectors from file.

    Parameters
    ---------
        path : str
            File path.
        size : int
            Number of vectors to load.
        expected_dimensions : int
            Expected number of dimensions in
            the laoded vectors.
        expected_dtype : str
            Expected datatype of each dimension
            of the loaded vectors.
        skip_first_line : bool
            Skip the first line because in
            many cases the first line of the vector
            file just tells the number of vectors.

    Returns
    ---------
        vectors : list
            Vectors as np array.
        words : list
            Matching array of words.
    """
    if skip_first_line:
        start_line = 1
        size += 1

    lines_range = range(start_line, size)

    words = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for i in tqdm(lines_range):
            try:
                # Get data from the line.
                line = f.readline().split()
                if expected_dtype == "bool_":
                    # Load bool values by upacking bits.
                    vector = np.unpackbits(
                        np.asarray(line[1:], dtype=np.uint8).reshape(1, -1)
                    )                   

                else:
                    # Otherwise just use the dtype.
                    vector = np.asarray(line[1:], dtype=expected_dtype).reshape(1, -1)[
                        0
                    ]

                if vector.shape[0] == expected_dimensions:
                    vectors.append(vector)
                    words.append(line[0])

            except ValueError as e:
                # Skip vectors which cannot be parsed.
                pass

        return tuple(map(np.array, (vectors, words)))


def convert_vectors_to_dict(
    vectors: list,
    words: list,
) -> dict:
    """
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
    """
    vectors_dict = {}
    for vector, word in zip(vectors, words):
        vectors_dict[word] = vector

    return vectors_dict


def save_vectors(
    path: str, words: list, vectors_batched: np.array, compression: str = "int8"
) -> None:
    """
    Save compressed word vectors to file provided
    aligned corpus of words and their corresponding vectors.

    Parameters
    ---------
        path : str
            Path to save the vectors.
        words : list
            Aligned list of vector words.
        vectors_batched : list
            List of batches of encoded vectors.
    """
    vectors = []
    for batch in vectors_batched:
        # Determine batch size dynamically because
        # the final batch may not meet the full size.
        for i in range(batch.shape[0]):
            vectors.append(batch[i])

    assert len(words) == len(vectors), "The encoded vectors could not be extracted."

    print("Exporting compressed vectors ...")
    with open(path, "w") as f:
        # Save vectors to new line with tab separating word and vector values.
        for word, vector in zip(words, vectors):
            try:
                vector = np.packbits(vector.astype("bool_"))
                f.write(word + "\t")
                f.write("\t".join(str(num) for num in vector))
                f.write("\n")

            except Exception:
                pass
