import tensorflow as tf
from datasets.vectors.classification.iris import get_data
from examples.fully_connected.template import train
from tensorflow.keras import regularizers


def train_small_fc_relu():
    name = "small_fc_relu"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_tanh():
    name = "small_fc_relu"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.tanh, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_relu_reg():
    name = "small_fc_relu_reg"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,), kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)
