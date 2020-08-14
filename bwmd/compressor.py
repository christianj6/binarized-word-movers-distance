import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
# Silence warnings.
tf.get_logger().setLevel('ERROR')


class Encoder(tf.keras.layers.Layer):
    '''
    Encode input into lower dimension 'code' representation,
    for decoding by decoder layer.
    '''
    def __init__(self, reduced_dimensions, compression):
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
    def get_compression_function(compression):
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

        else:
            raise Exception('Compression dtype unsupported.')


    def call(self, input_, transform=False):
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
    def __init__(self, reduced_dimensions, original_dimensions):
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


    def call(self, code):
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
    def __init__(self, reduced_dimensions, original_dimensions, compression):
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


    def call(self, input_):
        '''
        Pass input through the layers.
        '''
        # Produce code.
        code = self.encoder(input_)
        # Reconstruct the input.
        reconstructed = self.decoder(code)

        # Return reconstruction for loss calculations.
        return reconstructed


def reconstruction_loss(model, input_):
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
        reconstruction_loss : float
            Loss evaluation.
    '''
    return tf.reduce_mean(tf.square(tf.subtract(model(input_), input_)))


def train(loss, model, optimizer, input_):
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
                        reduced_dimensions:int=60, compression:str='bool_'):
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


    def fit(self, vectors, epochs:int=20, batch_size:int=75):
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
        plt.savefig('res\\loss.png')
        # Store batch size for evaluation.
        self.batch_size = batch_size


    @staticmethod
    def get_learning_rate():
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
    def prepare_vectors(vectors, batch_size):
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
        # Shuffle.
        prepared_vectors = prepared_vectors.shuffle(vectors.shape[0],
                                        reshuffle_each_iteration=True)
        # Prefetch for faster computation.
        prepared_vectors = prepared_vectors.prefetch(batch_size * 4)

        return prepared_vectors


    def evaluate(self, vectors):
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


    def transform(self, path, expected_dimensions, save=True):
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
        # Path to export transformed vectors.
        export_path = f'res\\{path}_compressed_'
        # Get length of file.
        lines = 0
        with open(path, 'r') as f:
            for line in f:
                lines +=1

        # TODO: Get words.
        # TODO: Don't shuffle vectors.

        # Load all vectors from file.
        vectors = load_vectors(path, lines, expected_dimensions)
        # Cast vectors into tensorflow object.
        vectors = self.prepare_vectors(vectors, self.batch_size)
        # Iterate through batches, extracting only the encoded vectors.
        for step, batch in enumerate(vectors):
            code = self.autoencoder.encoder(batch, transform=True)
            print(code.shape)
            # TODO: Iterate through vectors and save to file.

        # TODO: Print statistics about reduction.
        # print(str(sys.getsizeof(input_.numpy()) / 75 * 8), '>>', str(sys.getsizeof(latent.astype('int8')) / 75 * 8))


    def save(self, path):
        '''
        Save trained model to file.

        Parameters
        ---------
            path : str
                Export location.
        '''
        # TODO: Save self.autoencoder to path with pickle
        pass


def load_vectors(path, size, expected_dimensions):
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

    Returns
    ---------
        vectors : list
            Vectors as np.array objects.
    '''
    vectors = []
    with open(path, 'r') as f:
        # Show computation updates.
        for i in tqdm(range(size)):
            try:
                # Get only vectors; first item is the word itself.
                vector = np.asarray(f.readline().split()[1:], dtype='float32').reshape(1, -1)
                # Error handling to prevent ragged array.
                if vector.shape[1] == expected_dimensions:
                    vectors.append(vector)
            except ValueError as e:
                # Skip vectors which cannot be cast to floats.
                pass

    return vectors
