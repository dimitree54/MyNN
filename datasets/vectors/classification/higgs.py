# https://www.tensorflow.org/datasets/catalog/higgs
import tensorflow_datasets
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 1000000


def get_data(batch_size):
    raw_train = tensorflow_datasets.load('higgs', split='train[:90%]')
    raw_val = tensorflow_datasets.load('higgs', split='train[10%:]')
    train_batches = raw_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = raw_val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches
