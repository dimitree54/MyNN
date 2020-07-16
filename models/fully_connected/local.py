import tensorflow as tf
from tensorflow.python.ops import gen_math_ops, nn


def no_local_loss(activated_weights, inputs, outputs):  # noqa
    return 0


def calc_local_loss_v1(activated_weights, inputs, outputs):
    mismatch = tf.abs(activated_weights - inputs)
    mismatch = tf.reduce_mean(mismatch, -1)
    local_loss = abs(outputs) * mismatch
    return local_loss


def calc_local_loss_v2(activated_weights, inputs, outputs):
    match = gen_math_ops.mat_mul(inputs, activated_weights)
    mismatch = 1 / match
    local_loss = abs(outputs) * mismatch
    return local_loss


def calc_local_loss_v3(activated_weights, inputs, outputs):
    match = gen_math_ops.mat_mul(inputs, activated_weights)
    mismatch = 1 - match / inputs.shape[-1]
    local_loss = abs(outputs) * mismatch
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
        if self.activation is not None:
            return self.activation(outputs)
        self.add_loss(self.local_loss_fn(activated_weights, inputs, outputs), inputs=True)
        return outputs
