from tensorflow.keras.layers import Dense
import pandas as pd
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Dense, GRU, Dropout, Activation
import tensorflow as tf  # Importing TensorFlow for the custom loss function
import numpy as np


Two_D = True
# raw_data and labels are data and labels
path = './'
if Two_D:
    raw_data = pd.read_csv(path + 'measurementCov.csv',  usecols=['raw_x', 'raw_y'])
    labels = pd.read_csv(path + 'measurementCov.csv',  usecols=['truth_azimuth', 'truth_range'])
    raw_data_testing = pd.read_csv(path + 'measurementCov_testing.csv',  usecols=['raw_x', 'raw_y'])
    labels_testing = pd.read_csv(path + 'measurementCov_testing.csv',  usecols=['truth_azimuth', 'truth_range'])
    Dim_train_raw = raw_data.shape[1]
    Dim_train_label = labels.shape[1]
    num_train = len(raw_data)
sequence_length = 20
batch_size = 100
dimension = 2


class CustomModel(Model):
    def __init__(self, num_x_signals):
        super(CustomModel, self).__init__()
        self.gru_layer = GRU(units=800,
                             return_sequences=False,
                             kernel_regularizer=regularizers.l2(0.4),
                             recurrent_regularizer=regularizers.l2(0.4)
                             )
        self.dropout_layer = Dropout(0.5)
        self.activation_layer = Activation('sigmoid')
        self.output_layer = Dense(dimension*2)

    def call(self, inputs):
        x = self.gru_layer(inputs)
        x = self.dropout_layer(x)
        x = self.activation_layer(x)
        output = self.output_layer(x)
        return output


num_x_signals = dimension
model = CustomModel(num_x_signals=num_x_signals)


def custom_loss(model_input, y_true, model_output, epoch):
    print(model_input.shape)
    print(model_output)
    model_output = tf.reshape(model_output, [-1, dimension*2])  # Reshape to 2D
    model_input = tf.reshape(model_input, [-1, dimension])  # Reshape to 2D
    model_input = tf.cast(model_input, tf.float32)
    unlabeled_data, y_pred = tf.split(model_output, [dimension, dimension], axis=1)
    batch_size = tf.shape(model_output)[0]
    sequence_length = tf.shape(model_input)[0] // batch_size
    model_input_reshaped = tf.reshape(model_input, [-1, sequence_length, 2])  # Adjust 20 to your sequence length
    # Extract the last vector from each batch
    last_vector_each_batch = model_input_reshaped[:, -1, :]
    if Two_D:
        # Cartesian coordinates from y_true
        x, y = tf.unstack(y_pred + last_vector_each_batch, axis=-1)
        # x, y = tf.unstack(y_pred, axis=-1)
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
        # Conversion to polar coordinates
        r = tf.sqrt(x ** 2 + y ** 2)
        phi = tf.atan2(x, y) * (180.0 / tf.constant(np.pi))  # azimuth
        # Stacking the polar coordinates back into one tensor
        y_pred_polar = tf.stack([phi, r], axis=-1)
    else:
        # Cartesian coordinates from y_true
        x, y, z = tf.unstack(y_true, axis=-1)
        x, y, z = tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(z, tf.float32)
        # Conversion to polar coordinates
        r = tf.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = tf.asin(z / (r + 1e-8)) * (180.0 / tf.constant(np.pi))  # elevation
        phi = tf.atan2(x, y) * (180.0 / tf.constant(np.pi))  # azimuth
        # Stacking the polar coordinates back into one tensor
        y_true_polar = tf.stack([phi, theta, r], axis=-1)  # shape will be (batch_size, 3)

    # Reshape y_true to match the shape of y_pred
    y_true_reshaped = tf.reshape(y_true, tf.shape(y_pred_polar))
    y_true_reshaped = tf.cast(y_true_reshaped, dtype=tf.float32)
    # Ensure unlabeled_data is positive (since it's used as a diagonal matrix for a covariance matrix)
    unlabeled_data = tf.math.abs(unlabeled_data)
    squared_elements = tf.square(unlabeled_data)
    # Compute error e_i for each element
    error = y_pred_polar - y_true_reshaped
    if epoch < 0:
        positive_loss = tf.reduce_mean(tf.square(y_pred_polar - y_true_reshaped))
    else:
        # Create diagonal matrices G_i for all elements in the batch
        R = tf.linalg.diag(squared_elements)
        # Compute the inverse of each matrix in R
        R_inv = tf.linalg.inv(R)
        # Compute the log determinant of G_batch
        log_det_G_batch = tf.linalg.logdet(R)  # shape: (batch_size,)

        # Expand dimensions of error to solve the linear system in a batched manner
        error_expanded = tf.expand_dims(error, axis=-1)

        # Solve the linear system for all elements in the batch
        solution_batch = tf.linalg.solve(R_inv, error_expanded)

        # Squeeze the extra dimension from solution_batch and error
        solution_batch_squeezed = tf.squeeze(solution_batch, axis=-1)
        error_squeezed = tf.squeeze(error_expanded, axis=-1)

        # Compute the dot product for the second term
        second_term_batch = tf.reduce_sum(error_squeezed * solution_batch_squeezed, axis=1)

        # Sum the terms for all elements in the batch
        loss_batch = log_det_G_batch + second_term_batch

        # Average the loss over all elements in the batch
        loss = tf.reduce_mean(loss_batch)
        positive_loss = tf.math.softplus(loss)

    return positive_loss


optimizer = tf.keras.optimizers.Adam()


def validation_step(data, labels, epoch):
    model_output = model(data)
    loss = custom_loss(data, labels, model_output, epoch)
    model_output = tf.reshape(model_output, [-1, dimension*2])
    unlabeled_data, test_result = tf.split(model_output, [dimension, dimension], axis=1)
    return loss, test_result, unlabeled_data


def train_step(data, labels, epoch):
    with tf.GradientTape() as tape:
        model_output = model(data)
        loss = custom_loss(data, labels, model_output, epoch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    model_output = tf.reshape(model_output, [-1, dimension*2])  # Reshape to 2D
    unlabeled_data, _ = tf.split(model_output, [dimension, dimension], axis=1)
    return loss, unlabeled_data


def train(generator, steps_per_epoch, epochs, validation_data):
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            x_batch, y_batch = next(generator)  # Get the next batch of data
            loss, unlabeled_data = train_step(x_batch, y_batch, epoch)  # Train on this batch
        validation_loss, validation_result, test_covariance = validation_step(*validation_data, epoch)
        # Combine all batches' unlabeled_data into a single array
        all_unlabeled_data_in_epoch = np.stack(test_covariance, axis=0)
        # Compute and print the average of each column
        column_means = np.mean(all_unlabeled_data_in_epoch, axis=0)
        print(f'Epoch {epoch + 1}, Training Loss: {loss.numpy()}, Validation Loss: {validation_loss:.4f}, '
              f'Cov: {column_means}')


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, Dim_train_raw)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        # y_shape = (batch_size, sequence_length, dimension)
        y_shape = (batch_size, dimension)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            idx = np.random.randint(num_train - sequence_length)
            # Copy the sequences of data starting at this index.
            x_batch[i] = raw_data[idx:idx + sequence_length]
            # y_batch[i] = labels[idx:idx + sequence_length]
            y_batch[i] = labels[idx + sequence_length-1:idx + sequence_length]

        yield (x_batch, y_batch)


# test data
# Convert raw_data and labels to tensors, gru expect ndim=3 input
raw_data_testing = tf.expand_dims(raw_data_testing, axis=1)  # Adds a time dimension of length 1
labels_testing = tf.expand_dims(labels_testing, axis=1)  # Adds a time dimension of length 1
test_data = (raw_data_testing, labels_testing)
# Create the generator
generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)

# Train the model
train(generator, steps_per_epoch=50, epochs=200, validation_data=test_data)

model.save('model_covNstate', save_format='tf')

# Load the saved model
model = tf.keras.models.load_model('model_covNstate', custom_objects={'custom_loss': custom_loss})

new_data = pd.read_csv(path + 'measurementCov_testing.csv',  usecols=['raw_x', 'raw_y'])


def create_sequences(data, window_size, step_size):
    sequences = []
    for i in range(0, len(data) - window_size + 1, step_size):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)


def make_predictions(model, data, sequence_length, step_size):
    num_samples = len(data)
    predictions = []

    for i in range(0, num_samples - sequence_length + 1, step_size):
        sequence = data[i:i + sequence_length]
        prediction = model.predict(sequence)
        predictions.append(prediction[-1])

    return np.array(predictions)


step_size = 1
sequence_length = 20

new_data_reshaped = create_sequences(new_data.values, sequence_length, step_size)
predictions = make_predictions(model, new_data_reshaped, sequence_length, step_size)

# Converting predictions to a DataFrame and saving to a CSV file
predictions_df = pd.DataFrame(predictions,
                              columns=['Unlabeled_1', 'Unlabeled_2', 'Predicted_1', 'Predicted_2'])
predictions_df.to_csv('predictions.csv', index=False)


