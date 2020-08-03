import tensorflow as tf
from datasets.vectors.classification.iris import get_data
from examples.fully_connected.template import train, fake_loss_object

from models.fully_connected.local import LocalFCLayer, LocalFCLayerWithExternalOutput, FCWithActivatedWeights, \
    build_backward_local_loss, build_combined_local_loss, calc_local_loss_v4, calc_local_loss_v3, calc_local_loss_v2, \
    calc_local_loss_v1

num_epochs = 100
batch_size = 32
lr = 0.01


def train_small_fc_relu():
    name = "small_fc_relu"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr)


def train_small_fc_tanh():
    name = "small_fc_tanh"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.tanh, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(3, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr)


def train_small_fc_tanh_activated_kernel():
    name = "small_fc_tanh_activated_kernel"
    model = tf.keras.Sequential([
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh, input_shape=(4,)),
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh),
        FCWithActivatedWeights(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr)


def train_small_fc_tanh_activated_kernel_local_start(version="v4", bidirectional=False, no_grad=False):
    name = "small_fc_tanh_activated_kernel_local_start_" + version

    if version == "v1":
        local_loss = calc_local_loss_v1
    elif version == "v2":
        local_loss = calc_local_loss_v2
    elif version == "v3":
        local_loss = calc_local_loss_v3
    elif version == "v4":
        local_loss = calc_local_loss_v4
    else:
        local_loss = None

    local_losses = [local_loss]
    if bidirectional:
        name += "_bidirectional"
        local_losses += build_backward_local_loss(local_loss)
    else:
        name += "_forward"

    if no_grad:
        name += "_no_grad"

    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh, input_shape=(4,),
                     local_loss_fn=build_combined_local_loss(local_losses), stop_input_gradients=no_grad),
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh,
                     local_loss_fn=build_combined_local_loss(local_losses), stop_input_gradients=no_grad),
        FCWithActivatedWeights(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr)


def train_small_fc_tanh_activated_kernel_local_end(version="v4", bidirectional=False, no_grad=False):
    name = "small_fc_tanh_activated_kernel_local_start_" + version

    if version == "v1":
        local_loss = calc_local_loss_v1
    elif version == "v2":
        local_loss = calc_local_loss_v2
    elif version == "v3":
        local_loss = calc_local_loss_v3
    elif version == "v4":
        local_loss = calc_local_loss_v4
    else:
        local_loss = None

    local_losses = [local_loss]
    if bidirectional:
        name += "_bidirectional"
        local_losses += build_backward_local_loss(local_loss)
    else:
        name += "_forward"

    if no_grad:
        name += "_no_grad"

    model = tf.keras.Sequential([
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh, input_shape=(4,)),
        FCWithActivatedWeights(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh),
        LocalFCLayerWithExternalOutput(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh,
                                       local_loss_fn=build_combined_local_loss(local_losses),
                                       stop_input_gradients=no_grad)
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr, loss_object=fake_loss_object)


def train_small_fc_tanh_activated_kernel_fully_local(version="v4", bidirectional=False, no_grad=False):
    name = "small_fc_tanh_activated_kernel_local_start_" + version

    if version == "v1":
        local_loss = calc_local_loss_v1
    elif version == "v2":
        local_loss = calc_local_loss_v2
    elif version == "v3":
        local_loss = calc_local_loss_v3
    elif version == "v4":
        local_loss = calc_local_loss_v4
    else:
        local_loss = None

    local_losses = [local_loss]
    if bidirectional:
        name += "_bidirectional"
        local_losses += build_backward_local_loss(local_loss)
    else:
        name += "_forward"

    if no_grad:
        name += "_no_grad"

    model = tf.keras.Sequential([
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh, input_shape=(4,),
                     local_loss_fn=build_combined_local_loss(local_losses), stop_input_gradients=no_grad),
        LocalFCLayer(10, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh,
                     local_loss_fn=build_combined_local_loss(local_losses), stop_input_gradients=no_grad),
        LocalFCLayerWithExternalOutput(3, kernel_activation=tf.nn.tanh, activation=tf.nn.tanh,
                                       local_loss_fn=build_combined_local_loss(local_losses),
                                       stop_input_gradients=no_grad),
    ])
    train_data, val_data = get_data(batch_size)
    train(model, name, train_data, val_data, num_epochs, lr, loss_object=fake_loss_object)


train_small_fc_relu()
