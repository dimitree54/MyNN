from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Sequential, Model


class ConvBn(Sequential):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__([
            Conv2D(filters, kernel_size, stride, use_bias=False),
            BatchNormalization()
        ], **kwargs)


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

    def __init__(self, bottleneck_filters, final_filters, stride, conv_block=ConvBn, activation_block=ReLU):
        super().__init__()
        self.conv_block = conv_block
        # TODO do we need projection for stride=1? Also I do not like 1x1 convolution with stride 2.
        self.conv1 = conv_block(bottleneck_filters, 1, stride)
        self.conv1_activation = activation_block()
        self.conv2 = None
        self.conv2_activation = activation_block()
        self.conv3 = conv_block(final_filters, 1, 1)

        self.projection = conv_block(final_filters, 1, stride)
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


class ClassificationHead(Sequential):
    def __init__(self, num_classes):
        super().__init__([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes)
        ])


class ResNetBackbone(Model):
    down_stride = 2

    def __init__(self, nf, num_repeats: List[int],
                 return_endpoints_on_call=False,
                 init_block=InitialBlock,
                 main_block=ResNetIdentityMappingBlock,
                 down_block=ResNetDownBlock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_endpoints_on_call = return_endpoints_on_call

        filters = nf * 4
        bottleneck_filters = nf
        self.blocks = [
            [
                init_block(bottleneck_filters),
                down_block(bottleneck_filters, filters, 1)
                # TODO we use this block to change num filters before ResNet. Is it correct?
            ]
        ]

        for n in num_repeats:
            block = []
            for _ in range(n):
                block.append(main_block(bottleneck_filters))

            filters *= 2
            bottleneck_filters *= 2

            block.append(down_block(bottleneck_filters, filters, self.down_stride))
            self.blocks.append(block)

    def call(self, inputs, training=None, mask=None):
        endpoints = []
        x = inputs
        for block in self.blocks:
            for layer in block:
                x = layer(x, training, mask)
            endpoints.append(x)
        if self.return_endpoints_on_call:
            return endpoints
        else:
            return endpoints[-1]


def get_resnet50_backbone(nf):
    return ResNetBackbone(nf, [2, 4, 6, 2])


def get_resnet101_backbone(nf):
    return ResNetBackbone(nf, [2, 4, 23, 2])


def get_resnet152_backbone(nf):
    return ResNetBackbone(nf, [2, 8, 36, 2])
