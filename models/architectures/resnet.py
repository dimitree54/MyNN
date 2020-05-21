from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Sequential, Model


class ConvBn(Sequential):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__([
            Conv2D(filters, kernel_size, stride, use_bias=False),
            BatchNormalization()
        ])


class InitialBlock(Sequential):
    kernel_size = 7
    pool_size = 3
    stride = 2

    def __init__(self, filters, conv_block=ConvBn, activation_block=ReLU, pool_block=MaxPool2D):
        super().__init__([
            conv_block(filters, self.kernel_size, self.stride),
            activation_block(),
            pool_block(self.pool_size, self.stride)
        ])


class ResNetIdentityMappingBlock(Model):
    kernel_size = 3

    def __init__(self, bottleneck_filters, conv_block=ConvBn, activation_block=ReLU):
        super().__init__()
        self.conv_block = conv_block

        self.conv1 = conv_block(bottleneck_filters, 1, 1)
        self.conv1_activation = activation_block()
        self.conv2 = conv_block(bottleneck_filters, self.kernel_size, 1)
        self.conv2_activation = activation_block()
        self.conv3 = None
        self.sum_block = tf.keras.layers.Add()
        self.sum_activation = activation_block()

    def build(self, input_shape):
        self.conv3 = self.conv_block(input_shape[-1], 1, 1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x, training, mask)
        x = self.conv1_activation(x, training, mask)
        x = self.conv2(x, training, mask)
        x = self.conv2_activation(x, training, mask)
        x = self.conv3(x, training, mask)
        x = self.sum_block([x, inputs], training, mask)
        x = self.sum_activation(x, training, mask)
        return x


class ResNetDownBlock(Model):
    kernel_size = 3

    def __init__(self, bottleneck_depth, final_depth, stride, conv_block=ConvBn, activation_block=ReLU):
        super().__init__()
        self.conv_block = conv_block
        # TODO do we need projection for stride=1? Also I do not like 1x1 convolution with stride 2.
        self.conv1 = conv_block(bottleneck_depth, 1, stride)
        self.conv1_activation = activation_block()
        self.conv2 = None
        self.conv2_activation = activation_block()
        self.conv3 = conv_block(final_depth, 1, 1)

        self.projection = conv_block(final_depth, 1, stride)
        self.sum_block = tf.keras.layers.Add()
        self.sum_activation = activation_block()

    def build(self, input_shape):
        self.conv2 = self.conv_block(input_shape[-1], self.kernel_size, 1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x, training, mask)
        x = self.conv1_activation(x, training, mask)
        x = self.conv2(x, training, mask)
        x = self.conv2_activation(x, training, mask)
        x = self.conv3(x, training, mask)
        x = self.sum_block([x, self.projection(inputs)], training, mask)
        x = self.sum_activation(x, training, mask)
        return x


class ResNet:
    def __init__(self, num_repeats: List[int],
                 init_block=InitialBlock,
                 main_block=ResNetIdentityMappingBlock,
                 down_block=ResNetDownBlock):
        pass
