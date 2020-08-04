# https://www.tensorflow.org/datasets/catalog/german_credit_numeric
# one class classification, input has 24 int features, pos/neg=630/270
import tensorflow_datasets
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 1000


def get_data(batch_size):
    raw_train = tensorflow_datasets.load('german_credit_numeric', split='train[:90%]')  # 0.9 * 1000
    raw_val = tensorflow_datasets.load('german_credit_numeric', split='train[10%:]')  # 0.1 * 1000
    train_batches = raw_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = raw_val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches
