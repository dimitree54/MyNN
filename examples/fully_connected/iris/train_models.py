import tensorflow as tf
from datasets.vectors.classification.iris import get_data
from examples.fully_connected.template import train
from tensorflow.keras import regularizers

from models.fully_connected.local import LocalFCLayer, calc_local_loss_v1, calc_local_loss_v2, calc_local_loss_v3


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
    name = "small_fc_tanh"
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


def train_small_fc_local_v1():
    name = "small_fc_local_v1"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.keras.activations.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_local_v2():
    name = "small_fc_local_v2"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v2,
                     activation=tf.keras.activations.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v2,
                     activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_local_v3():
    name = "small_fc_local_v3"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v3,
                     activation=tf.keras.activations.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.keras.activations.tanh, local_loss_fn=calc_local_loss_v3,
                     activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


train_small_fc_relu_reg()
train_small_fc_tanh()
train_small_fc_relu()
train_small_fc_local_v1()
train_small_fc_local_v2()
train_small_fc_local_v3()
