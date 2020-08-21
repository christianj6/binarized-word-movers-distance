import unittest
from bwmd.compressor import Compressor, load_vectors


# TODO: helper functions to prepare the word similarity task

class TestCase(unittest.TestCase):
    '''
    '''
    def test_compressor_functionality(self):
        '''
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
        pass


    def test_compressor_speed(self):
        '''
        '''
        pass


###

# TODO: Introduce early stopping when it converges.
# TODO: Confirm the gradient error can be ignored.
# TODO: Confirm batch size is appropriate.
# TODO: Confirm dimensions are appropriate given the (n*m notation in the paper.)
# TODO: Identify optimal training hyperparameters.
# TODO: Confirm activation functions are appropriate.
# TODO: Add more updates.
# TODO: Adjust reconstruction loss to custom function from Tissier et al.
# TODO: Evaluate binary embeddings with test script.
# TODO: Clean, document, comment, annotate.

###
