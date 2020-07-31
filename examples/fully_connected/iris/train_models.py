import tensorflow as tf
from datasets.vectors.classification.iris import get_data
from examples.fully_connected.template import train, fake_loss_object

from models.fully_connected.local import LocalFCLayer, calc_local_loss_v1, \
    LocalFCLayerWithExternalOutput, FCWithActivatedWeights, build_backward_local_loss, \
    build_combined_local_loss, calc_local_activity_normalization_loss, calc_local_loss_v4


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
        tf.keras.layers.Dense(3, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_tanh_activated_kernel():
    name = "small_fc_tanh_activated_kernel"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh,
                               input_shape=(4,)),  # input shape required
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh),
        FCWithActivatedWeights(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_sigmoid_activated_kernel():
    name = "small_fc_sigmoid_activated_kernel"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        FCWithActivatedWeights(10, kernel_activation=tf.nn.sigmoid, activation=tf.nn.sigmoid,
                               input_shape=(4,)),  # input shape required
        FCWithActivatedWeights(10, kernel_activation=tf.nn.sigmoid, activation=tf.nn.sigmoid),
        FCWithActivatedWeights(3, kernel_activation=tf.nn.sigmoid, activation=tf.nn.sigmoid)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_tanh_activated_kernel_local_start():
    name = "small_fc_tanh_activated_kernel_local_start"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.nn.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.nn.tanh),
        FCWithActivatedWeights(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01)


def train_small_fc_tanh_activated_kernel_local_end():
    name = "small_fc_tanh_activated_kernel_local_end_v4_bidirectional"
    num_epochs = 2010
    batch_size = 32
    model = tf.keras.Sequential([
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh,
                               activation=tf.nn.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.nn.tanh,
                     local_loss_fn=build_combined_local_loss([
                         calc_local_activity_normalization_loss
                     ]),
                     activation=tf.nn.tanh),
        LocalFCLayerWithExternalOutput(3, kernel_activation=tf.nn.tanh,
                                       local_loss_fn=build_combined_local_loss([
                                           calc_local_loss_v4,
                                           build_backward_local_loss(calc_local_loss_v4),
                                           # calc_local_activity_normalization_loss
                                       ]),
                                       activation=tf.nn.tanh),
    ])
    train_data, val_data = get_data(batch_size)  # TODO    V    reduced lr for debug
    train(model, name, train_data, val_data, num_epochs, 0.001, loss_object=fake_loss_object)


def train_small_fc_tanh_activated_kernel_fully_local():
    name = "small_fc_tanh_activated_kernel_fully_local"
    num_epochs = 201
    batch_size = 32
    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.nn.tanh, input_shape=(4,)),
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, local_loss_fn=calc_local_loss_v1,
                     activation=tf.nn.tanh),
        LocalFCLayerWithExternalOutput(3, kernel_activation=tf.nn.tanh,
                                       local_loss_fn=None,  # TODO choose loss
                                       activation=tf.nn.tanh),
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, 0.01, loss_object=fake_loss_object)


train_small_fc_tanh_activated_kernel_local_end()
