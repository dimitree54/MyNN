import tensorflow as tf
from tensorflow.python.ops import gen_math_ops, nn


def no_local_loss(activated_weights, inputs, activated_outputs):  # noqa
    return 0


def calc_local_loss_v1(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    _activated_weights = tf.expand_dims(activated_weights, 0)  # [1, n, m]
    _inputs = tf.expand_dims(inputs, -1)  # [bs, n, 1]
    mismatch = tf.abs(_activated_weights - _inputs)  # [bs, n, m]
    mismatch = tf.reduce_mean(mismatch, 1)  # [bs, m]
    local_loss = abs(activated_outputs) * mismatch  # [bs, m]
    local_loss = tf.reduce_mean(local_loss)  # []
    return local_loss


def calc_local_loss_v2(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    match = gen_math_ops.mat_mul(inputs, activated_weights)  # [bs, m]
    mismatch = 1 / match  # [bs, m]
    local_loss = abs(activated_outputs) * mismatch  # [bs, m]
    local_loss = tf.reduce_mean(local_loss)  # []
    return local_loss


def calc_local_loss_v3(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    match = gen_math_ops.mat_mul(inputs, activated_weights)  # [bs, m]
    mismatch = 1 - match / inputs.shape[-1]  # [bs, m]
    local_loss = abs(activated_outputs) * mismatch  # [bs, m]
    local_loss = tf.reduce_mean(local_loss)  # []
    return local_loss


class LocalFCLayer(tf.keras.layers.Dense):
    def __init__(self, units,
                 kernel_activation=None,
                 local_loss_fn=calc_local_loss_v1, **kwargs):
        self.local_loss_fn = local_loss_fn
        self.kernel_activation = kernel_activation
        super().__init__(units, **kwargs)

    def call(self, inputs):
        if self.kernel_activation is None:
            activated_weights = self.kernel
        else:
            activated_weights = self.kernel_activation(self.kernel)
        outputs = gen_math_ops.mat_mul(inputs, activated_weights)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is None:
            activated_outputs = outputs
        else:
            activated_outputs = self.activation(outputs)
        self.add_loss(self.local_loss_fn(activated_weights, inputs, activated_outputs), inputs=True)
        return activated_outputs
