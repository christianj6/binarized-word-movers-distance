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

| FastText-EN (512-bit) | GloVe-EN (512-bit) | Word2Vec-EN (512-bit) |
| --------------------- | ------------------ | --------------------- |
| [Download]()          | [Download]()       | [Download]()          |

***

### Usage

- [ ] minimal start
- [ ] creating a new model
- [ ] info on all parameters you can tweak

***

### Tests

Tests are implemented with the unittest built-in library. You may run all tests from the repository directory with the following command.

```python -m unittest bwmd```

Specific tests may be run by accessing the *test* module.

```$ python -m unittest bwmd.test.test_triplets```

***

### TODO

- [ ] Information on obtaining and formatting vectors.
- [ ] make sure all the demo code works

***

### References

- Kusner, Matt & Sun, Y. & Kolkin, N.I. & Weinberger, Kilian. (2015). From word embeddings to document distances. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). 957-966.
