# Binarized Word Mover's Distance
Scalable text similarity with binarized embedding distance and syntactic dependency information.

***

### Overview

This repository hosts code for the thesis 'Binarized Word Mover's Distance,' written by Christian Johnson for obtaining the M.Sc. in Cognitive Science at Osnabrück University. The project concerns the enhancement of a preexisting distance metric, the Word Mover's Distance, originally developed by Kusner et al. (2015). 

***

### Project Structure

<pre>
|-- bwmd
|-- res
	|-- datasets
	|-- images
	|-- models
|-- test
</pre>

##### bwmd

Main source code directory, including scripts for autoencoder vector compression and distance calculations.

##### res

Resources and outputs. *datasets* contains the word similarity data used to evaluate compressed word vectors. *images* contains output images including a summary of autoencoder training. *models* contains saved models in case this parameter is provided during fitting. This directory is the expected location of the vectors used to fit the autoencoder model. As the vectors are too large to host on GitHub, they must be downloaded and manually-placed into this directory.

##### test

Test scripts, including several evaluations for the binary vectors themselves and the distance metric.

***

### Installation

Please install the library with pip by first cloning the repository and navigating to the repository directory.

<pre>$ pip install setup.py</pre>

***

### Tests

Tests are implemented with the unittest built-in library. You may run all tests from the repository directory with the following command.

<pre>python -m unittest</pre>

Specific tests may be run by accessing the *test* module.

<pre>$ python -m unittest test.test_compressor</pre>

***

### TODO

- [ ] Module for distance calculations.
- [ ] Test for distance metric evaluations.
- [ ] Information on obtaining and formatting vectors.
- [ ] CLI support.

***

### References

- Kusner, Matt & Sun, Y. & Kolkin, N.I. & Weinberger, Kilian. (2015). From word embeddings to document distances. Proceedings of the 32nd International Conference on Machine Learning (ICML 2015). 957-966. 