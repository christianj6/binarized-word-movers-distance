import unittest
from bwmd.compressor import Compressor, load_vectors


# TODO: Helper functions to prepare the word similarity task

class TestCase(unittest.TestCase):
    '''
    Test cases for autoencoder compressor
    for reduced memory footprint word vectors.
    '''
    def test_compressor_functionality(self):
        '''
        Test of basic compressor functionality.
        '''
        # Vectors filepath.
        path = 'res\\glove.840B.300d.txt'
        # Vector dimension.
        dim = 300
        # Load vectors.
        vectors = load_vectors(path, 200_000, dim)
        # Split into training and validation.
        train, validation = vectors[:150_000], vectors[150_000:]
        # Instantiate compressor and fit to training data.
        compressor = Compressor(original_dimensions=dim,
                                    reduced_dimensions=30,
                                        compression='int8')
        compressor.fit(train, epochs=3)
        # Evaluate loss on validation set.
        compressor.evaluate(validation)
        # Transform and save all original vectors.
        compressor.transform(path, dim, save=True)
        # Save model.
        compressor.save('res\\models\\glove_compressor')


    def test_compressor_word_similarity(self):
        '''
        Test performance of compressed vectors on the
        semantic word similarity task cf. Tissier (2018).
        '''
        # TODO: All evaluation test cases.
        pass


    def test_compressor_speed(self):
        '''
        Test to compare the speed of compressed
        vectors against that of original vectors. Use the
        knn task on the entire vector space.
        '''
        pass
