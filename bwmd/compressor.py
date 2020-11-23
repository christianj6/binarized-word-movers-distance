'''
OVERVIEW
This module contains classes and methods for the
autoencoder compression of pretrained word vectors.
Autoencoder methods are implemented in Tensorflow
and wrapped in a meta-class for handling
model training and vector transformation.

USAGE
Create an instance of the Compressor meta class
and first fit() the object on a set of
pretrained word vectors, contained in a .txt file
as words and vector values separated by a tab
character, separated by newlines. Inspect the
Compressor class to determine the additional
parameters needed for training, transformation.
After training, you can transform the same set of
vectors or another using the transform() method, which
allows one to immmediately save the transformed vectors
to a file.
'''
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from bitstring import BitArray
# Silence warnings.
tf.get_logger().setLevel('ERROR')
from gmpy2 import pack


class Encoder(tf.keras.layers.Layer):
    '''
    Encode input into lower dimension 'code' representation,
    for decoding by decoder layer.
    '''
    def __init__(self, reduced_dimensions:int, compression:str)->None:
        '''
        Initialize layers.

        Parameters
        ---------
            reduced_dimensions : int
                Size of intermediate 'code'
                representation. Should be of appropriate
                size that the 8-bit boolean values ultimately
                fit within CPU registers.
            compression : str
                Name of numpy dtype to which we wish to compress
                each of the input values when reduced to the
                code representation. Should be as small as possible
                i.e. 8-bits.
        '''
        super(Encoder, self).__init__()
        # Prepare compressor and compression function.
        self.compression = compression
        self.compression_fn = self.get_compression_function(compression)
        # Tensorflow handles input size dynamically so we set the number
        # of units to the intermediate representation size for
        # compatibility with the output layer.
        self.hidden_layer1 = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            )
        self.hidden_layer2 = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            )
        self.output_layer = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.sigmoid
            )


    @staticmethod
    def get_compression_function(compression:str)->'Function':
        '''
        Factory for compression lambda functions, which
        conditionally-reduce encoded float values to
        a smaller range which can be represented
        with lower-memory data types.

        Parameters
        ---------
            compression : str
                each of the input values when reduced to the
                code representation. Should be as small as possible
                i.e. 8-bits.

        Returns
        ---------
            compression_fn : lambda function
                Function to conditionally reduce float values
                to a smaller range.
        '''
        if compression == 'bool_':
            # Function to binarize the latent representation.
            return lambda x: np.array([[0 if i <= 0.5 else 1 for i in y] for y in x])

        elif compression == 'int8':
            # Function for linear rescaling of values to within int8 range.
            mx, mn = 1, 0
            return lambda x: ((x - mn) / (mx - mn)) * (127 - -128) + -128
            # return lambda x: ((x - np.amin(x)) / (np.amax(x) - np.amin(x))) * (127 - -128) + -128

        else:
            raise Exception('Compression dtype unsupported.')


    def call(self, input_:np.array, transform:bool=False)->'tf.Tensor':
        '''
        Pass input through the layer.
        '''
        activation = self.hidden_layer1(input_)
        activation = self.hidden_layer2(activation)
        # Get output of encoder.
        output = self.output_layer(activation)
        # Convert output to compressed code.
        latent = self.compression_fn(output.numpy())
        if transform:
            # If only transforming, immediately return the code.
            return latent.astype(self.compression)

        return tf.convert_to_tensor(latent.astype(self.compression))


class Decoder(tf.keras.layers.Layer):
    '''
    Decode intermediate 'code' representation, attempting
    to reproduce the original encoder input.
    '''
    def __init__(self, reduced_dimensions:int,
                        original_dimensions:int=300)->None:
        '''
        Initialize layers.

        Parameters
        ---------
            reduced_dimensions : int
                Size of intermediate 'code'
                representation. Should be of appropriate
                size that the 8-bit boolean values ultimately
                fit within CPU registers.
            original_dimensions : int
                Size of original input vector.
        '''
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            )
        # Output layer is same size as the original vector.
        # Here we use sigmoid for continuous float values.
        self.output_layer = tf.keras.layers.Dense(
                units=original_dimensions,
                activation=tf.nn.sigmoid
            )


    def call(self, code:'tf.Tensor')->'tf.Tensor':
        '''
        Pass input through the layer.
        '''
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class AutoEncoder(tf.keras.Model):
    '''
    Encoder-decoder architecture for producing
    compressed vector representations.
    '''
    def __init__(self, reduced_dimensions:int=30,
                    original_dimensions:int=300, compression:str='int8')->None:
        '''
        Initialize encoder and decoder layers.

        Parameters
        ---------
            reduced_dimensions : int
                Size of intermediate 'code'
                representation. Should be of appropriate
                size that the 8-bit boolean values ultimately
                fit within CPU registers.
            original_dimensions : int
                Size of original input vector.
            compression : str
                Name of numpy dtype to which we wish to compress
                each of the input values when reduced to the
                code representation. Should be as small as possible
                i.e. 8-bits.
        '''
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(reduced_dimensions, compression)
        self.decoder = Decoder(reduced_dimensions, original_dimensions)


    def call(self, input_:np.array)->'tf.Tensor':
        '''
        Pass input through the layers.
        '''
        # Produce code.
        code = self.encoder(input_)
        # Reconstruct the input.
        reconstructed = self.decoder(code)

        # Return reconstruction for loss calculations.
        return reconstructed


@tf.function
def get_rank(y_pred):
    '''
    Helper function for spearman correlation.

    Parameters
    ---------
        y_pred : tf.Tensor
            Predicted values

    Returns
    ---------
        y_pred (ranked) : tf.Tensor
            Sorted tensor.
    '''
    # Start ranking at 1 instead of zero.
    rank = tf.argsort(tf.argsort(y_pred, axis=-1, direction="ASCENDING"), axis=-1) + 1
    return y_pred


@tf.function
def sp_rank(x, y):
    '''
    Calculate the spearman rank correlation
    between a pair of samples.

    Parameters
    ---------
        x : tf.Tensor
            Single sample.
        y : tf.Tensor
            Single sample.

    Returns
    ---------
        score : tf.Tensor
            Correlation score.
    '''
    cov = tfp.stats.covariance(x, y, sample_axis=0, event_axis=None)
    sd_x = tfp.stats.stddev(x, sample_axis=0, keepdims=False, name=None)
    sd_y = tfp.stats.stddev(y, sample_axis=0, keepdims=False, name=None)

    # Return 1-score because loss function is for minimization.
    return 1-cov/(sd_x*sd_y)


@tf.function
def spearman_correlation(y_true, y_pred):
    '''
    Calculate the spearman correlation score
    between two tensors. Used for calculating a custom loss
    function.

    Parameters
    ---------
        y_true : tf.Tensor
            Ground-truth matrix to which we wish
            to determine the degree of correlation.
        y_pred : tf.Tensor
            Predicted values for comparison.

    Returns
    ---------
        loss : tf.Tensor
            Loss value as spearman correlation score
            for the two input tensors.
    '''
    # Obtain a ranking of the predicted values.
    y_pred_rank = tf.map_fn(lambda x: get_rank(x), y_pred, dtype=tf.float32)

    #Spearman rank correlation between each pair of samples.
    sp = tf.map_fn(lambda x: sp_rank(x[0],x[1]), (y_true, y_pred_rank), dtype=tf.float32)
    #Reduce to a single value
    loss = tf.reduce_mean(sp)
    return loss


@tf.function
def pairwise_distance(feature: 'TensorLike', squared: bool = False):
    '''
    Compute pairwise distance matrix for an input tensor. Used
    for spearman correlation loss function.

    Parameters
    ---------
        feature : tf.Tensor
            Input tensor.

    Returns
    ---------
        pairwise_distances : tf.Tensor
            2-D Tensor with pairwise distances.
    '''
    # Calculate the distances.
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(feature, tf.transpose(feature))
    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )
    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
    )
    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data])
    )
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def reconstruction_loss(model, input_:'tf.Tensor')->'tf.Tensor':
    '''
    Mean squared error loss.

    Parameters
    ---------
        model : tf.keras.Model
            Trained model.
        input_ : tf.tensor
            Original training input.

    Returns
    ---------
        reconstruction_loss : tf.Tensor
            Loss evaluation as float cast
            as tf.Tensor.
    '''
    # Compute standard reconstruction loss.
    reconstruction_loss = tf.reduce_mean(tf.square(tf.subtract(model(input_), input_)))
    # Compute spearman correlation coefficient.
    spearman = spearman_correlation(pairwise_distance(input_), pairwise_distance(model(input_)))
    # Convert spearman to similarity measure.
    spearman = 1 - spearman

    for v in model.trainable_variables:
        # Extract weights for encoder hidden layer.
        if v.name == 'auto_encoder/encoder/dense/kernel:0':
            w = v
            wt = tf.transpose(w)
            # Get n*n identity matrix.
            i = tf.eye(w.numpy().shape[1])
            # Compute linear normalization.
            norm = tf.norm(tf.linalg.matmul(wt, w) - i)
            # Regularization parameter lambda.
            lambda_reg = 4
            # Compute regularization parameter l.
            l_reg = tf.square(norm) / 2
            # Compute final loss equation.
            loss = reconstruction_loss + lambda_reg * (l_reg + spearman)

            return loss


def train(loss, model:'tf.Keras.Model',
                optimizer:'tf.optimizers.Optimizer', input_:'tf.Tensor')->None:
    '''
    Train model by applying gradients to trainable parameters.

    Parameters
    ---------
        loss : method
            Loss function.
        model : tf.keras.Model
            Trained model.
        optimizer : tf.optimizer
            Gradient optimizer.
        input_ : tf.tensor
            Raw training input.
    '''
    # Gradient tape for persistent data.
    with tf.GradientTape() as tape:
        # Compute gradients.
        gradients = tape.gradient(loss(model, input_), model.trainable_variables)

    variables = zip(gradients, model.trainable_variables)
    # Apply gradients.
    optimizer.apply_gradients(variables)


class Compressor():
    '''
    Meta class for training autoencoder and
    compressing a set of real-valued vectors
    according to the chosen compression format.
    '''
    def __init__(self, original_dimensions:int=300,
                        reduced_dimensions:int=60, compression:str='bool_')->None:
        '''
        Initialize autoencoder with appropriate parameters.

        Parameters
        ---------
            reduced_dimensions : int
                Size of intermediate 'code'
                representation. Should be of appropriate
                size that the 8-bit boolean values ultimately
                fit within CPU registers.
            original_dimensions : int
                Size of original input vector.
            compression : str
                Name of numpy dtype to which we wish to compress
                each of the input values when reduced to the
                code representation. Should be as small as possible
                i.e. 8-bits.
        '''
        # Instantiate configured autoencoder.
        self.autoencoder = AutoEncoder(reduced_dimensions, original_dimensions, compression)


    def fit(self, vectors, epochs:int=20, batch_size:int=75)->None:
        '''
        Fit the autoencoder to a set of training vectors.

        Parameters
        ---------
            vectors : list
                Training vectors as np.array objects.
            epochs : int
                Number of training epochs.
            batch_size : int
                Training batch size.
        '''
        # Get learning rate schedule.
        learning_rate = self.get_learning_rate()
        # Optimizer.
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        # Cast vectors into tf.Dataset object.
        training = self.prepare_vectors(vectors, batch_size)

        losses = []
        # Train for n epochs.
        for epoch in range(epochs):
            print('Epoch: ', str(epoch), end='\t\t')
            for step, batch in enumerate(training):
                train(reconstruction_loss, self.autoencoder, optimizer, batch)
                loss = reconstruction_loss(self.autoencoder, batch)

            # Store loss after each epoch.
            losses.append(loss)
            print('Loss: ', str(round(loss.numpy(), 3)))

        # Save a summary of training.
        plt.plot(losses)
        # plt.savefig('res\\images\\loss.png')
        plt.show()
        # Store batch size for evaluation.
        self.batch_size = batch_size


    @staticmethod
    def get_learning_rate()->'tf.optimizers.schedule.Schedule':
        '''
        Generate a learning rate schedule to improve
        training stability during later steps.

        Returns
        ---------
            learning_rate : tf.keras.optimizers.LearningRateSchedule
                Decaying learning rate schedule.
        '''
        # Step boundaries after which the learning rate value changes.
        boundaries = [9000]
        # Sequential values for piecewise learning rate change.
        values = [1e-5, 1e-7]

        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)


    @staticmethod
    def prepare_vectors(vectors:list, batch_size:int, shuffle=True)->'tf.Tensor':
        '''
        Cast vectors into tensorflow dataset object
        for batching and/or shuffling for improved
        training performance.

        Parameters
        ---------
            vectors : list
                Vectors as np.array objects.
            batch_size : int
                Training batch size to which the
                model was fitted.
            shuffle : bool
                Shuffle vectors or not. Needed for use in
                additional helper functions where
                shuffling is inappropriate.

        Returns
        ---------
            prepared_vectors : tf.data.Dataset
                Tensorflow compatible dataset.
        '''
        # Cast all into a single array.
        vectors = np.array(vectors).astype('float32')
        # Remove extraneous dimension.
        vectors = np.squeeze(vectors, axis=1)
        # Convert to tf dataset.
        prepared_vectors = tf.data.Dataset.from_tensor_slices(vectors)
        # Batch.
        prepared_vectors = prepared_vectors.batch(batch_size)
        if shuffle:
            # Shuffle.
            prepared_vectors = prepared_vectors.shuffle(vectors.shape[0],
                                            reshuffle_each_iteration=True)
        # Prefetch for faster computation.
        prepared_vectors = prepared_vectors.prefetch(batch_size * 4)

        return prepared_vectors


    def evaluate(self, vectors:list)->None:
        '''
        Evaluate the trained model against
        unseen validation data.

        Parameters
        ---------
           vectors : list
                Vectors as np.array objects.
        '''
        # Throw exception if model is not fitted.
        assert self.batch_size, 'Model must be fit before evaluation.'
        # Prepare validation vectors.
        validation = self.prepare_vectors(vectors, self.batch_size)
        losses = []
        # Iterate through batches without training.
        for step, batch in enumerate(validation):
            loss = reconstruction_loss(self.autoencoder, batch)
            losses.append(loss.numpy())

        # Return an average of the validation losses.
        print('\nValidation loss: ', str(round(sum(losses) / len(losses), 3)))


    def transform(self, path:str, expected_dimensions:int=300, save=True)->None:
        '''
        Transform real-valued vectors to a compressed code
        representation using a trained autoencoder. Option to
        save the compressed vectors to a file.

        Parameters
        ---------
            path : str
                Location of vectors to be transformed.
            expected_dimensions : int
                Expected size of real-valued vectors. Needed
                to avoid creating a ragged array when there are
                errors in the original data.
            save : bool
                Save transformed vectors to file.
        '''
        # Create a folder to save compressed vectors.
        output_dir = f"{''.join(path.split('.')[0:-1])}"
        os.makedirs(output_dir, exist_ok=True)
        # Path to export transformed vectors.
        export_path = f'{output_dir}\\vectors.txtc'
        # Load all vectors and words from file.
        vectors, words = load_vectors(path, expected_dimensions=expected_dimensions, get_words=True)
        # Get size of original vector before turning it into a tf dataset.
        size_original_vector = sys.getsizeof(vectors[0]) * 8
        # Cast vectors into tensorflow object. Do not shuffle
        # to preserve compatibility with words.
        vectors = self.prepare_vectors(vectors, self.batch_size, shuffle=False)
        # Iterate through batches, extracting only the encoded vectors.
        vectors_encoded = []
        # Get compression for saving vectors.
        compression = self.autoencoder.encoder.compression
        print('Encoding vectors ...')
        for step, batch in enumerate(vectors):
            code = self.autoencoder.encoder(batch, transform=True)
            vectors_encoded.append(code)

        if save:
            save_vectors(export_path, words, vectors_encoded, compression=compression)

        # Get size of encoded vector.
        size_encoded_vector = round(sys.getsizeof(vectors_encoded[0]) / self.batch_size * 8)
        # Print statistics about memory reduction.
        print(f'Vectors of size {size_original_vector} bits reduced to {size_encoded_vector} bits.')


    def save(self, path:str)->None:
        '''
        Save trained model to file.

        Parameters
        ---------
            path : str
                Export location.
        '''
        # Use tf function because pickling of
        # dynamic objects is unsupported.
        self.autoencoder.save_weights(path)


    def load(self, path:str)->None:
        '''
        Load trained model from file.

        Parameters
        ---------
            path : str
                Import location.
        '''
        self.autoencoder.load_weights(path)


def save_vectors(path:str, words:list, vectors_batched:np.array, compression:str='int8')->None:
    '''
    Save compressed word vectors to file provided
    aligned corpus of words and their corresponding vectors.

    Parameters
    ---------
        path : str
            Path to save the vectors.
        words : list
            Aligned list of vector words.
        vectors_batched : list
            List of batches of encoded vectors.
    '''
    vectors = []
    for batch in vectors_batched:
        # Determine batch size dynamically because
        # the final batch may not meet the full size.
        for i in range(batch.shape[0]):
            vectors.append(batch[i])

    assert len(words) == len(vectors), 'The encoded vectors could not be extracted.'

    print('Exporting compressed vectors ...')
    with open(path, 'w') as f:
        # Save vectors to new line with tab separating word and vector values.
        for word, vector in zip(words, vectors):
            if compression == 'int8':
                f.write(word + '\t')
                f.write('\t'.join(str(num) for num in vector))
                f.write('\n')
            elif compression == 'bool_':
                vector = vector.astype(int)
                f.write(word + '\t')
                f.write(''.join(str(num) for num in vector))
                f.write('\n')


def load_vectors(path,
    size:int=None,
    expected_dimensions:int=300,
    expected_dtype:str='float32',
    get_words:bool=False,
    bitarray:bool=False,
    return_numpy:bool=False,
    skip_first_line:bool=False
)->list:
    '''
    Load word embedding vectors from file.

    Parameters
    ---------
        path : str
            Location of real-valued vectors.
        size : int
            Number of vectors to extract.
        expected_dimensions : int
            Expected size of real-valued vectors. Needed
            to avoid creating a ragged array when there are
            errors in the original data.
        expected_dtype : str
            Expected data type of loaded vectors.
        words : bool
            Whether or not to return the original
            words with the vectors.
        skip_first_line : bool
            Skip the first line in cases where
            pretrained vectors have a header.

    Returns
    ---------
        vectors : list
            Vectors as np.array objects.
        words : list
            Words aligned with vectors.
    '''
    # If no specific size provided, get length.
    if size == None:
        size = 0
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                size +=1

    if skip_first_line:
        start_line = 1
        size += 1
    else:
        start_line = 0
    lines_range = range(start_line, size)

    words = []
    vectors = []
    with open(path, 'r', encoding="utf8") as f:
        # Show computation updates.
        for i in tqdm(lines_range):
            try:
                line = f.readline().split()
                # Get word.
                word = line[0]
                if expected_dtype == 'bool_':
                    bits = line[1]
                    if bitarray:
                        # Option to use bitarray representation, which
                        # allows us to retrieve the original string,
                        # necessary for interface with some
                        # external libraries.
                        vector = BitArray(bin=bits)
                        if len(bits) == expected_dimensions:
                            vectors.append(vector)
                            words.append(word)
                        continue

                    elif return_numpy:
                        # Option to use numpy array, necessary to take
                        # advantage of numba jit compiling, which does
                        # not support other objects.
                        vector = np.asarray(list(bits), dtype='int8').reshape(1, -1)
                        if len(bits) == expected_dimensions:
                            vectors.append(vector)
                            words.append(word)
                        continue

                    # Otherwise convert vector to integers and pack into
                    # primitive data structure, which is much faster.
                    vector = list(bits)
                    vector = [int(item) for item in vector]
                    # Pack vector into primitive. The resulting object will
                    # not have the same length as the original
                    # vector but will remain an accurate bitwise
                    # representation for hamming calculations.
                    vector = pack(vector,1)
                    if len(bits) == expected_dimensions:
                        vectors.append(vector)
                        words.append(word)
                    continue
                elif expected_dtype == 'int8':
                    vector = np.asarray(line[1:], dtype='int8').reshape(1, -1)
                else:
                    vector = np.asarray(line[1:], dtype='float32').reshape(1, -1)

                # Error handling to prevent ragged array.
                if vector.shape[1] == expected_dimensions:
                    vectors.append(vector)
                    words.append(word)
            except ValueError as e:
                # Skip vectors which cannot be cast to floats.
                continue

    # Alternative return for words.
    if get_words:
        return vectors, words

    return vectors
