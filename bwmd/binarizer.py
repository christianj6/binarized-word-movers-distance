import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###

# TODO: Adjust encoder output activation to heaviside for binary.
# TODO: Confirm batch size is appropriate.
# TODO: Identify optimal training hyperparameters via grid search.
# TODO: Confirm activation functions are appropriate.
# TODO: Make dimensions more dynamic.
# TODO: Add training updates.
# TODO: Adjust reconstruction loss to custom function from Tissier et al.
# TODO: Binarizer class to handle fit, transform, save output, save class.
# TODO: Evaluate binary embeddings with test script.
# TODO: Clean, document, comment, annotate.

###


class Encoder(tf.keras.layers.Layer):
    '''
    Encode input into lower dimension 'code' representation,
    for decoding by decoder layer.
    '''
    def __init__(self, reduced_dimensions):
        '''
        Initialize layers.

        Parameters
        ---------
            reduced_dimensions : int
                Size of intermediate 'code'
                representation. Should be of appropriate
                size that the 8-bit boolean values ultimately
                fit within CPU registers.
        '''
        super(Encoder, self).__init__()
        # Tensorflow handles input size dynamically so we set the number
        # of units to the intermediate representation size for
        # compatibility with the output layer.
        self.hidden_layer = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.relu,
                kernel_initializer='he_uniform'
            )
        self.output_layer = tf.keras.layers.Dense(
                units=reduced_dimensions,
                activation=tf.nn.sigmoid
            )


    def call(self, input_):
        '''
        Pass input through the layer.
        '''
        activation = self.hidden_layer(input_)
        return self.output_layer(activation)


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
    def __init__(self, reduced_dimensions, original_dimensions):
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
        '''
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(reduced_dimensions)
        self.decoder = Decoder(reduced_dimensions, original_dimensions)


    def call(self, input_):
        '''
        Pass input through the layers.
        '''
        # Produce code.
        code = self.encoder(input_)
        # Reconstruct the input.
        reconstructed = self.decoder(code)

        # Return reconstruction attempt for loss calculations.
        return reconstructed


def reconstruction_loss(model, input_):
    '''
    Custom loss function for evaluating the reconstructions
    produced by the decoder. Based on Tissier et al. (2019).

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


if __name__ == '__main__':
    # Training hyperparameters.
    epochs = 20
    batch_size = 300#128
    learning_rate = 1e-5#1e-3
    original_dimensions = 300
    reduced_dimensions = 50

    # Instantiate autoencoder and optimizer.
    autoencoder = AutoEncoder(reduced_dimensions, original_dimensions)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Get real-value word embedding vectors from Glove.
    features = []
    with open('glove.840B.300d.txt', 'r') as f:
        for i in range(100_000):
            try:
                feature = np.asarray(f.readline().split()[1:], dtype='float32').reshape(1, -1)
                features.append(feature)
            except ValueError as e:
                # Skip vectors which cannot be cast to floats.
                print(e)
                pass

    # Cast all into a single array.
    features = np.array(features).astype('float32')
    # Remove extraneous dimension.
    features = np.squeeze(features, axis=1)
    # Convert to tf dataset.
    training = tf.data.Dataset.from_tensor_slices(features)
    # Batch.
    training = training.batch(batch_size)
    # Shuffle.
    training = training.shuffle(features.shape[0])
    # Prefetch for faster computation.
    training = training.prefetch(batch_size * 4)

    losses = []
    # Train for n epochs.
    for epoch in range(epochs):
        for step, batch in enumerate(training):
            train(reconstruction_loss, autoencoder, optimizer, batch)
            loss = reconstruction_loss(autoencoder, batch)

        # Store loss after each epoch.
        losses.append(loss)

    # Visualize training losses.
    plt.plot(losses)
    plt.savefig('loss.png')
