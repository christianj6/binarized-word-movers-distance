# old code to inform the new OO approach


def hamming_distance(self):
    """
    Decorator to return
    hamming distance between
    two word vectors.
    """

    def distance(a: str, b: str):
        """
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
        """
        return np.count_nonzero(self.vectors[a] != self.vectors[b]) / self.dim

    return distance


def default_maximum_value(self, default: float = 1.0):
    """
    Decorator for returning a default
    maximum value.
    """

    def distance(a: str, b: str):
        """
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
        """
        return default

    return distance


def related_word_distance_lookup(self):
    """
    Decorator to lookup the distance between related
    words using a precomputed
    lookup table of cosine distances. If the
    value cannot be found, function returns
    None.
    """

    def distance(a: str, b: str):
        """
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
        """
        try:
            # Try lookup.
            return self.cache[a][b]

        except (KeyError, AttributeError):
            # Otherwise None for error handling.
            return None

    return distance


@staticmethod
def get_pairwise_distance_matrix(
    text_a: list, text_b: list, dist: "Function", default: "Function"
) -> "np.array":
    """
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
    """
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


def get_rwmd(self, text_a: list, text_b: list):
    """
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
    """
    # Pairwise distance matrix.
    pdist = self.get_pairwise_distance_matrix(
        text_a,
        text_b,
        # Euclidean distance.
        dist=lambda a, b: np.sqrt(
            np.sum((self.vectors[a] - self.vectors[b]) ** 2)
        ),
        # Default nan value.
        default=lambda a, b: 1,
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

    # TODO: Multiprocessing for large corpora.


@staticmethod
def min_distance_unidirectional(pdist: "np.array"):
    """
    Get minimum distance in one direction.

    Parameters
    ---------
        pdist : np.array
            Pairwise distance matrix.

    Returns
    ---------
        distance : float
            RWMD.
    """
    d = 0
    for i in pdist:
        d += min(i)

    return d


def get_relrwmd(self, text_a: list, text_b: list):
    """
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
    """
    # Pairwise distance matrix.
    pdist = self.get_pairwise_distance_matrix(
        text_a,
        text_b,
        # Precomputed lookup cf Werner.
        dist=self.related_word_distance_lookup(),
        # Default maximum value cf Werner (cMax-value)
        default=self.default_maximum_value(default=1.0),
    )
    relrwmd = self.min_distance_unidirectional(pdist)
    # Transpose to get other direction.
    relrwmd += self.min_distance_unidirectional(pdist.T)

    return relrwmd
