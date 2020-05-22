"""
This file contains activation function blocks. We want all blocks to implement tf.keras.Model call interface,
so some functions is just wrapping keras functions and layers into keras.Model.
Unified call interface simplifies blocks replacements.
"""
import tensorflow as tf


class ReLU(tf.keras.Sequential):
    def __init__(self):
        super().__init__([tf.keras.layers.ReLU()])


class Mish(tf.keras.Sequential):
    """
    https://arxiv.org/abs/1908.08681
    smooth alternative for relu
    """
    def __init__(self):
        super().__init__([
            tf.keras.layers.Activation(self.mish)
        ])

    @staticmethod
    def mish(x):
        return x * tf.tanh(tf.math.log(1 + tf.exp(x)))
