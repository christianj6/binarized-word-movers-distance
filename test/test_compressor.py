import unittest
from cwmd.compressor import Compressor, load_vectors
from cwmd.distance import convert_vectors_to_dict
import pandas as pd
from scipy.spatial import distance
# from sklearn.metrics import jaccard_score
from scipy import stats
import random


REDUCED_DIMENSIONS = 256
COMPRESSION = 'bool_'
MODELS_TO_TEST = ['glove', 'fasttext', 'word2vec']

class TestCase(unittest.TestCase):
    '''
    Test cases for autoencoder compressor
    for reduced memory footprint word vectors.
    '''
    def test_compressor_functionality(self):
        '''
        Test of basic compressor functionality.
        '''
        for name in MODELS_TO_TEST:
            # Vectors filepath.
            path = f'res\\{name}.txt'
            # Vector dimension.
            dim = 300
            # Load vectors.
            vectors = load_vectors(path, 300_000, dim)
            # Split into training and validation.
            train, validation = vectors[:250_000], vectors[250_000:]
            # Instantiate compressor and fit to training data.
            compressor = Compressor(original_dimensions=dim,
                                        reduced_dimensions=REDUCED_DIMENSIONS,
                                            compression=COMPRESSION)
            compressor.fit(train, epochs=10)
            # Evaluate loss on validation set.
            compressor.evaluate(validation)
            # Transform and save all original vectors.
            compressor.transform(path, dim, save=True)
            # Save model.
            compressor.save(f'res\\models\\{name}\\{name}_compressor')


    def test_compressor_word_similarity(self):
        '''
        Test performance of compressed vectors on the
        semantic word similarity task cf. Tissier (2018).
        '''
        def score_vectors_on_similarity_task(datasets, vectors):
            '''
            Main function for scoring on similarity tasks.

            Parameters
            ---------
                datasets : list
                    Names of datasets for which we want to
                    evaluate the vectors.
                vectors : str
                    Name of file containing the vectors for
                    testing both the original and compressed
                    versions
            '''
            def calculate_word_similarity_score(data, vectors_dict):
                '''
                Calculate the spearmen rank correlation
                coefficient to evaluate the accuracy of vectors
                against ground-truth data.

                Parameters
                ---------
                    data : pd.DataFrame
                        Ground truth data as dataframe with
                        lexical comparisions and human similarity
                        scores.
                    vectors_dict : dict
                        Dictionary mapping words to numerical
                        vectors.
                '''
                human_similarities = []
                hamming_similarities = []
                for _, row in data.iterrows():
                    word1 = row['Word 1']
                    word2 = row['Word 2']
                    # Append the ground-truth similarities.
                    human_similarities.append(float(row['Similarity']))
                    # Get hamming distance for binary vectors.
                    hamming_distance = distance.hamming(vectors_dict[word1], vectors_dict[word2])
                    # Subtract from one to convert to similarity score.
                    hamming_similarities.append(1 - hamming_distance)

                # Calculate spearman score.
                score,_ = stats.spearmanr(human_similarities, cosine_similarities)
                return score

            real_value = f'res\\{vectors}.txt'
            compressed = f'{real_value}c'
            # Evaluate both real and compressed vectors.
            vectors_to_test = [real_value, compressed]
            for vector_path in vectors_to_test:
                # Load vectors from file.
                vectors, words = load_vectors(vector_path, size=lines,
                                        get_words=True, expected_dimensions=REDUCED_DIMENSIONS,
                                        expected_dtype=COMPRESSION)
                # Convert to dict for easier access.
                vectors = convert_vectors_to_dict(vectors, words)
                for dataset in datasets:
                    # Load data as dataframe.
                    data = pd.read_csv(f'res\\datasets\\{dataset}.csv')
                    # Calculate and print the score.
                    score = calculate_word_similarity_score(data, vectors)
                    print('Spearman Correlation: ', str(round(score, 3)),
                                            f'({vector_path} + {dataset})')

                # TODO: Save a summary for UX.

        datasets = ['simverb3500', 'wordsim353']
        for vectors in MODELS_TO_TEST:
            score_vectors_on_similarity_task(datasets, vectors)


    def test_compressor_speed(self):
        '''
        Test to compare the speed of compressed
        vectors against that of original vectors. Use the
        knn task on the entire vector space.
        '''
        # TODO: Nested function so I can run with both.
        # TODO: Load vectors into memory as np.arrays.
        # TODO: Perform KNN on them per Werner.
        pass
