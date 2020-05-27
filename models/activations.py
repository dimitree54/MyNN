"""
This file contains activation function blocks.
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
    def __init__(self, **kwargs):
        super().__init__([
            tf.keras.layers.Activation(self.mish)
        ], **kwargs)

    @staticmethod
    def mish(x):
        return x * tf.tanh(tf.math.log(1 + tf.exp(x)))
