from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="bwmd",
    version="0.2.1",
    description="Fast text similarity with binary encoded word embeddings.",
    url="https://github.com/christianj6/binarized-word-movers-distance.git",
    author="Christian Johnson",
    author_email="",
    license="unlicensed",
    package_dir={"bwmd": "bwmd"},
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.6.0",
        "tensorflow_probability==0.13.0",
        "numpy==1.19.5",
        "tqdm==4.61.2",
        "scipy==1.7.1",
        "nltk==3.6.2",
        "pyemd==0.5.1",
        "gensim==3.8.3",
        "scikit-learn==0.24.2",
        "scipy==1.4.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
