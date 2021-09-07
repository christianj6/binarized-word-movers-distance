# Binarized Word Mover's Distance

Scalable text similarity with encoded embedding distance.

***

![BBC MDS](https://github.com/christianj6/binarized-word-movers-distance/raw/master/res/mds_bbc.png)

*Description: MDS visualization of [BBC News](http://mlg.ucd.ie/datasets/bbc.html) data. Dissimilarity matrix computed using Binarized Word Mover's Distance.*

***

### Overview

The Binarized Word Mover's Distance (BWMD) is a modification of the Word Mover's Distance, originally developed by Kusner et al. (2015). The BWMD computes a lower-bound Wasserstein word-embedding distance using binarized embeddings and an approximate-nearest-neighbor cache. 

[Paper](https://github.com/christianj6/binarized-word-movers-distance/raw/master/res/johnson_2020.pdf)

***

### Installation

```
pip install bwmd
```

***

### Models

To compute distances, you must provide a path to a model directory containing a compressed vector file and approximate-nearest-neighbor lookup tables. You can compute these yourself as described in the ```/notebooks/``` directory, or use one of the models below.

| FastText-EN (512-bit) | GloVe-EN (512-bit) | Word2Vec-EN (512-bit) |
| --------------------- | ------------------ | --------------------- |
| [Download]()          | [Download]()       | [Download]()          |

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

Sample code for this minimal start and for training your own compressed vectors for any language can be found in the ```/notebooks/``` directory.

***

### API Details

- ```bwmd.distance.BWMD(model_path, size_vocab, language, dim, raw_hamming=False)``` creates a distance object from a path containing precomputed lookup tables and compressed vectors. You must specify the total number of vocabulary items, language (for removing stopwords), and dimension of the compressed vectors. If you wish only to use the raw hamming distances and not lookup table values, specify ```raw_hamming=True```.
- ```bwmd.distance.BWMD().get_distance(text_a, text_b)``` computes the BWMD between two texts as lists of strings. This method assumes that stopwords have already been removed.
- ```bwmd.distance.BWMD().pairwise(corpus)``` computes a pairwise distance matrix for an array of texts as lists of strings.
- ```bwmd.distance.BWMD().preprocess_text(text)``` removes stopwords and out-of-vocabulary words from a single text as a list of strings.
- ```bwmd.distance.BWMD().preprocess_corpus(corpus)``` removes stopwords and out-of-vocabulary words from a corpus as an array of texts as lists of strings.
- ```bwmd.compressor.Compressor(original_dimensions, reduced_dimensions, compression)``` creates a compressor object which will accept word embeddings of dimension ```original_dimensions``` and compress them to dimension ```reduced_dimensions``` according to the data type specified in ```compression```. 
- ```bwmd.compressor.Compressor().fit(vectors, epochs=20, batch_size=75)``` fits the compressor to the input vectors by training an autoencoder under the specified hyperparameters.
- ```bwmd.compressor.Compressor().transform(path, n_vectors, save=False)``` transforms the original vectors residing at the specified path using a trained autoencoder. The ```n_vectors``` parameter specifies what amount of vectors, starting at the beginning of the vector file, will ultimately be transformed and returned. If ```save=True``` the transformed vectors will be saved to the input path.
- ```bwmd.tools.load_vectors(path, size, expected_dimensions, expected_dtype, skip_first_line=True)``` loads and returns vectors and words from a text file containing words and vector features on each new line. The parameter ```skip_first_line``` should be set to ```True```when the first line of a vector file is vector metadata and not an actual vector.
- ```bwmd.tools.convert_vectors_to_dict(vectors, words)``` casts aligned arrays of vectors and words into a Python dictionary with words as keys.
- ```bwmd.partition.build_partitions_lookup_tables(vectors, I, real_value_path, vector_dim)``` uses a special partitioning algorithm similar to bisecting k-means to identify approximate-nearest-neighbors for each input vector. The free parameter ```I``` controls the ```k``` number of partitions which are to be made, leading to ```k = 2^I``` partitions.

***

### Obtaining Real-Valued Vectors

To compute compressed vectors such as those provided above, you must provide a txt file containing words and vector features separated by newline characters. You can obtain high-quality vectors for many languages from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html). If using ```.vec``` files from FastText, ensure you set ```skip_first_line=True``` when loading vectors from a file. Further details on parsing real-valued vector files when training your own models can be found in the ```fit_model.ipynb``` example in the *notebooks* directory.

***

### References

- Kusner, Matt & Sun, Y. & Kolkin, N.I. & Weinberger, Kilian. (2015). From word embeddings to document distances. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). 957-966.
- Werner, Matheus & Laber, Eduardo. (2019). Speeding up Word Mover's Distance and its variants via properties of distances between embeddings. 

***

### Deployment

1. Debug local packaging.
   1. ```pip install -e .```
2. Build wheel.
   1. ```python setup.py bdist_wheel sdist```
3. Upload to PyPI.
   1. ```twine upload dist/*```

