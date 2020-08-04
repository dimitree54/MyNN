"""
https://www.tensorflow.org/datasets/catalog/higgs
This is one-class classification problem to discriminate "tau tau decay of a Higgs boson" event with background.
Data has 11 000 000 entries
Baseline network is 5-layer network with 300 neurons per layer with tanh activation
"""
import tensorflow_datasets
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 1000000


def get_data(batch_size):
    raw_train = tensorflow_datasets.load('higgs', split='train[:90%]')  # 0.9 * 11,000,000
    raw_val = tensorflow_datasets.load('higgs', split='train[10%:]')  # 0.1 * 11,000,000
    train_batches = raw_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = raw_val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches
