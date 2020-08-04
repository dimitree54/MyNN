import tensorflow as tf


def no_local_loss(activated_weights, inputs, activated_outputs):  # noqa
    return 0


def build_local_activity_normalization_loss(normal_activity_rate=0.3):
    def calc_local_activity_normalization_loss(activated_weights, inputs, activated_outputs):  # noqa
        # this normalization loss works with activated_outputs in range [-1, 1] (after tanh activation)
        normal_activity = activated_outputs.shape[-1] * normal_activity_rate
        normal_activity = tf.expand_dims(normal_activity, 0)  # adding batch dimension
        activity_normalization_loss = tf.square(tf.reduce_sum(tf.abs(activated_outputs), axis=-1) - normal_activity)
        activity_normalization_loss = tf.reduce_mean(activity_normalization_loss)
        return activity_normalization_loss * 10
    return calc_local_activity_normalization_loss


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
    local_loss = tf.reduce_sum(local_loss)  # []
    return local_loss


def calc_local_loss_v2(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    match = tf.matmul(inputs, activated_weights)  # [bs, m]
    mismatch = 1 / match  # [bs, m]
    local_loss = abs(activated_outputs) * mismatch  # [bs, m]
    local_loss = tf.reduce_sum(local_loss)  # []
    return local_loss


def calc_local_loss_v3(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    match = tf.matmul(inputs, activated_weights)  # [bs, m]
    mismatch = 1 - match / inputs.shape[-1]  # [bs, m]
    local_loss = abs(activated_outputs) * mismatch  # [bs, m]
    local_loss = tf.reduce_sum(local_loss)  # []
    return local_loss


def calc_local_loss_v4(activated_weights, inputs, activated_outputs):
    # n - input size, m - output size
    # activated_weights [n, m]
    # inputs [bs, n]
    # outputs [bs, m]
    match = tf.matmul(inputs, activated_weights)  # [bs, m]
    max_possible_match = tf.expand_dims(tf.reduce_sum(tf.abs(activated_weights), axis=0) + 0.0001, axis=0)  # [1, m]
    normalized_match = match / max_possible_match  # [bs, m]
    local_loss = tf.keras.losses.mean_squared_error(activated_outputs, normalized_match)  # [bs, m]
    # local_loss = tf.keras.losses.binary_crossentropy(activated_outputs, normalized_match)  # [bs, m]
    local_loss = tf.reduce_sum(local_loss)  # []
    return local_loss


def build_backward_local_loss(base_local_loss):
    def loss(activated_weights, inputs, activated_outputs):
        # n - input size, m - output size
        # activated_weights [n, m]
        # inputs [bs, n]
        # outputs [bs, m]
        # analogy with forward local loss:
        # inputs in backward = activated_outputs in forward
        # activated_outputs in backward = inputs in forward
        # activated_weights in backward = activated_weights^T
        backward_loss = base_local_loss(tf.transpose(activated_weights), activated_outputs, inputs)
        return backward_loss
    return loss


def build_combined_local_loss(local_losses: list):
    def loss(activated_weights, inputs, activated_outputs):
        total_loss = 0
        for local_loss in local_losses:
            total_loss += local_loss(activated_weights, inputs, activated_outputs)
        return total_loss
    return loss


class FCWithActivatedWeights(tf.keras.layers.Dense):
    def __init__(self, units, kernel_activation=None, **kwargs):
        self.kernel_activation = kernel_activation
        self.kernel = None
        super().__init__(units, **kwargs)

    def activate_kernel(self):
        self.kernel = self.kernel_activation(self.kernel)

    def call(self, inputs):
        temp = self.kernel
        self.activate_kernel()  # making hack with temp because otherwise kernel becomes not eager.
        output = super().call(inputs)
        self.kernel = temp
        return output


class LocalFCLayer(FCWithActivatedWeights):
    def __init__(self, units, local_loss_fn=None, stop_input_gradients=False, **kwargs):
        self.local_loss_fn = local_loss_fn
        self.stop_input_gradients = stop_input_gradients
        super().__init__(units, **kwargs)

    def call(self, inputs):
        if self.stop_input_gradients:
            inputs = tf.stop_gradient(inputs)
        activated_outputs = super().call(inputs)
        if self.local_loss_fn is not None:
            self.add_loss(self.local_loss_fn(self.kernel, inputs, activated_outputs), inputs=True)
        return activated_outputs


class LocalFCLayerWithExternalOutput(LocalFCLayer):
    def __init__(self, units, **kwargs):
        self.external_output = None
        super().__init__(units, **kwargs)
        # TODO output bias do not have gradients

    def call(self, inputs):
        if self.stop_input_gradients:
            inputs = tf.stop_gradient(inputs)  # stopping again specially for local_loss_fn
        temp = self.local_loss_fn
        self.local_loss_fn = None
        activated_outputs = super().call(inputs)  # skipping local loss in parents call
        self.local_loss_fn = temp
        if self.external_output is None:
            print("Warning no external output, so internal will be used")
            self.external_output = activated_outputs
        if self.local_loss_fn is not None:
            self.add_loss(self.local_loss_fn(self.kernel, inputs, self.external_output), inputs=True)
        self.external_output = None
        return activated_outputs

    def set_output(self, external_output):
        self.external_output = tf.cast(external_output, tf.float32)
