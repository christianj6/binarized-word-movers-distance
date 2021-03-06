from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='bwmd',
    version='0.1.0',
    description='Fast, robust text similarity with autoencoder compressed word embeddings.',
    url='https://github.com/christianj6/binarized-word-movers-distance.git',
    author='Christian Johnson',
    author_email='',
    license='unlicensed',
    package_dir={'bwmd': 'bwmd'},
    packages=find_packages(),
    install_requires=[
        'nltk==3.5',
        'numpy==1.18.5',
        'tensorflow==2.3.1',
        'tensorflow_probability==0.11.0',
        'tqdm==4.47.0',
        'gensim==3.8.3',
        'bitstring==3.1.7',
        'requests==2.24.0',
        'spacy==2.3.2',
        'networkx==2.5',
        'matplotlib==3.3.0',
        'pyemd==0.5.1',
        'dill==0.3.2',
        'scipy==1.4.1',
        'beautifulsoup4==4.9.3',
        'gmpy2==2.0.8',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
