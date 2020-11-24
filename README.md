# Binarized Word Mover's Distance

Scalable text similarity with binarized embedding distance and syntactic dependency information.

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

- [ ] info on all parameters you can tweak

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

Because many of the tests require real-valued and/or compressed vectors to function properly, it is impossible to comprehensively evaluate the test-suite without these files. Only by downloading all three of the above models and placing these in the package root directory along with GloVe vectors obtained [here](http://nlp.stanford.edu/data/glove.42B.300d.zip), can you safely run all tests. The GloVe vectors must be named ```glove.txt```. Additionally, to run evaluations on the Wikipedia triplets task to reproduce results seen in the paper, you must download the triplets data [here]() and place the unzipped folder in *bwmd/data/datasets*.

***

### TODO

- [ ] make sure all the demo code works

***

### References

- Kusner, Matt & Sun, Y. & Kolkin, N.I. & Weinberger, Kilian. (2015). From word embeddings to document distances. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). 957-966.
