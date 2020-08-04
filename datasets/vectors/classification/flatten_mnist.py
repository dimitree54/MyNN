# 9 classes classification
import tensorflow as tf
import numpy as np

SHUFFLE_BUFFER_SIZE = 10000


def get_data(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255) > 0.5
    x_train = np.reshape(x_train, (60000, -1)).astype(np.float)
    x_test = (x_train / 255) > 0.5
    x_test = np.reshape(x_test, (10000, -1)).astype(np.float)
    train_batches = tf.data.Dataset.from_tensor_slices({"features": x_train, "label": y_train}) \
        .shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = tf.data.Dataset.from_tensor_slices({"features": x_test, "label": y_test}) \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches
