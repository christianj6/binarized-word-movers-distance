# Binarized Word Mover's Distance

Scalable text similarity with binarized embedding distance and syntactic dependency information.

***

![BBC MDS](https://github.com/christianj6/binarized-word-movers-distance/raw/cleanup/mds_bbc.png)

*Description: MDS visualization of [BBC News](http://mlg.ucd.ie/datasets/bbc.html) data. Dissimilarity matrix computed using Binarized Word Mover's Distance.*

***

### Overview

The Binarized Word Mover's Distance (BWMD) is an adaption of the Word Mover's Distance, originally developed by Kusner et al. (2015). The BWMD computes a lower-bound Wasserstein word-embedding distance using binarized embeddings and an approximate-nearest-neighbor cache. 

[Paper]()

***

### Installation

Clone the repository.

```$ git clone https://github.com/christianj6/binarized-word-movers-distance.git```

Navigate to the repository directory and install with pip.

```$ pip install .```

***

### Models

In order to compute distances, you must provide a path to a model directory containing a compressed vector file and approximate-nearest-neighbor lookup tables. You can download the models used for evaluations from the following links.

| FastText-EN (512-bit)                                        | GloVe-EN (512-bit)                                           | Word2Vec-EN (512-bit)                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Download](https://drive.google.com/uc?export=download&id=1MSEltaeVk-mbzNGCbcfyXuHURqM5WRJt) | [Download](https://drive.google.com/uc?export=download&id=1xzVbGKV0fsuTCA9OkR5auJNgO05xCdAZ) | [Download](https://drive.google.com/uc?export=download&id=1M1Dd6RrWq8ZJk1l2zvf1YxFGuG-HNqxf) |

***

### Minimal Start

If you already possess a model directory for your language, you may quickly compute distances as in the example below.

```python
from bwmd.distance import BWMD

# Create a corpus of documents.
corpus = [
	['obama', 'speaks', 'to', 'the', 'media', 'in', 'illinois'],
	['the', 'president', 'greets', 'the', 'press', 'in', 'chicago'],
    	['this', 'sentence', 'is', 'unrelated']
]
# Instantiate a distance object.
bwmd = BWMD(
	model_path='glove-en-512',
	dim=512,
)
# Get pairwise distances.
bwmd.pairwise(corpus)
>>> array([[0.        , 0.23999023, 0.31298828],
           [0.23999023, 0.        , 0.31502279],
           [0.31298828, 0.31502279, 0.        ]])
```

Sample code for this minimal start and for training your own compressed vectors for any language can be found in the *notebooks* directory.

***

### API Details

- ```BWMD(model_path, dim, with_syntax=False, size_vocab=20_000)``` creates a distance object from a path containing precomputed lookup tables and compressed vectors. You must specify the dimension of the compressed vectors, otherwise the object will assume you are supplying real-valued vectors and you will not be able to compute the BWMD. 
- ```bwmd.get_distance(text_a, text_b)``` computes the BWMD between two texts as lists of strings. One should not remove stopwords as these are integral to the syntactic distance calculations and are automatically-removed.
- ```bwmd.pairwise(corpus)``` computes a pairwise distance matrix for an array of texts as lists of strings. 
- ```bwmd.get_wcd(text_a, text_b)``` computes the Word Centroid Distance for evaluative comparisons, per Kusner et al. (2015). When using this method one must create a BWMD object from real-valued vectors.
- ```bwmd.get_wmd(text_a, text_b)``` computes the Word Mover's Distance for evaluative comparisions, per Kusner et al. (2015). When using this method one must create a BWMD object from real-valued vectors.
- ```bwmd.get_rwmd(texta, text_b)``` computes the Relaxed Word Mover's Distance lower-bound for evaluative comparisons, per Kusner et al. (2015). When using this method one must create a BWMD object from real-valued vectors.
- ```bwmd.get_relrwmd(text_a, text_b)``` computes the Related Relaxed Word Mover's Distance lower-bound for evaluative comparisions, per Werner et al. (2019). This method is compatible with the standard BWMD model format but computes an alternative lower-bound without compressed vectors.
- ```Compressor(original_dimensions, reduced_dimensions, compression='bool_')``` creates an autoencoder compressor object which can be fitted to a set of real-valued word embeddings that they can be transformed into a lower-dimensional, binary space. 
- ```compressor.fit(vectors, epochs=20, batch_size=75)``` will train/fit the autoencoder on a set of loaded vectors. 
- ```compressor.transform(path, expected_dimensions, n_vectors=30_000)``` compresses and saves the supplied vectors to a corresponding model directory. With the ```n_vectors``` parameter you can control what amount of the original vector space is ultimately transformed. Given that most vector files are sorted by word frequency, it is often unnecessary to transform the full vector space and with this parameter you can save some computation time and on-disk memory.
- ```load_vectors(path, size, expected_dimensions, expected_dtype, get_words=False)``` loads vectors and/or words into a tuple of arrays and can be used to supply a compressor with vectors directly or instantiate a BWMD object when converted to a dictionary.
- ```convert_vectors_to_dict(vectors, words)``` zips vectors and words into a dictionary mapping tokens to their vector representations.
- ```build_partitions_lookup_tables(vectors, I, real_value_path, vector_size)``` constructs approximate-nearest-neighbor lookup tables for caching vector distances. The free parameter ```I``` controls the number of partitions which are made at each of the n=100 iterations of the algorithm, resulting in ```k=2^I``` partitions.

***

### Obtaining Real-Valued Vectors

To generate compressed vectors, you first need a file containing real-valued vectors as words/values separated by spaces and new lines. You can obtain high-quality vectors for many languages from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html). Make sure you download the .txt files and set ```skip_first_line=True``` when loading vectors from the file. Further details on parsing real-valued vector files when training your own models can be found in the ```fit_model.ipynb``` example in the *notebooks* directory.

***

### Tests

Tests are implemented with the unittest built-in library. You may run all tests from the repository directory with the following command.

```python -m unittest bwmd```

Specific tests may be run by accessing the *test* module.

```$ python -m unittest bwmd.test.test_triplets```

#### Note: Files Required for Running Tests

Because many of the tests require real-valued and/or compressed vectors to function properly, it is impossible to comprehensively evaluate the test-suite without these files. Only by downloading all three of the above models and placing these in the package root directory along with GloVe vectors obtained [here](http://nlp.stanford.edu/data/glove.42B.300d.zip), can you safely run all tests. The GloVe vectors must be named ```glove.txt```. Additionally, to run evaluations on the Wikipedia triplets task to reproduce results seen in the paper, you must download the triplets data [here](https://drive.google.com/uc?export=download&id=1dxSVO1t0mzHs5_mHklO0qaI2lDsCbhv3) and place the unzipped folder in *bwmd/data/datasets*.

***

### References

- Kusner, Matt & Sun, Y. & Kolkin, N.I. & Weinberger, Kilian. (2015). From word embeddings to document distances. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). 957-966.
- Werner, Matheus & Laber, Eduardo. (2019). Speeding up Word Mover's Distance and its variants via properties of distances between embeddings. 
