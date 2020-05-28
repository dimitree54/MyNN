from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras import Sequential, Model

from models.activations import ReLU


class ConvBn(Sequential):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__([
            Conv2D(filters, kernel_size, stride, use_bias=False, padding='same'),
            BatchNormalization()
        ], **kwargs)


class InitialBlock(Sequential):
    kernel_size = 7
    pool_size = 3
    stride = 2

    def __init__(self, filters,
                 conv_block: Model = ConvBn, activation_block: Model = ReLU, pool_block: Model = MaxPool2D, **kwargs):
        super().__init__([
            conv_block(filters, self.kernel_size, self.stride),
            activation_block(),
            pool_block(self.pool_size, self.stride, padding='same')
        ], **kwargs)


class ResNetBottleneckBlock(Model):
    kernel_size = 3

    def __init__(self, conv_block: Model = ConvBn, activation_block: Model = ReLU, **kwargs):
        super().__init__(**kwargs)
        self.conv_block = conv_block

        self.conv1 = None
        self.conv1_activation = activation_block()
        self.conv2 = None
        self.conv2_activation = activation_block()
        self.conv3 = None
        self.sum_block = tf.keras.layers.Add()
        self.sum_activation = activation_block()

    def build(self, input_shape):
        input_filters = input_shape[-1]
        bottleneck_filters = input_filters // 4
        self.conv1 = self.conv_block(bottleneck_filters, 1, 1)
        self.conv2 = self.conv_block(bottleneck_filters, self.kernel_size, 1)
        self.conv3 = self.conv_block(input_filters, 1, 1)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv1(x, **kwargs)
        x = self.conv1_activation(x, **kwargs)
        x = self.conv2(x, **kwargs)
        x = self.conv2_activation(x, **kwargs)
        x = self.conv3(x, **kwargs)
        x = self.sum_block([x, inputs])
        x = self.sum_activation(x, **kwargs)
        return x


class ResNetIdentityBlock(Model):
    def __init__(self, conv_block: Model = ConvBn, activation_block: Model = ReLU, **kwargs):
        super().__init__(**kwargs)
        self.conv_block = conv_block

        self.conv1 = None
        self.conv1_activation = activation_block()
        self.conv2 = None
        self.sum_block = tf.keras.layers.Add()
        self.sum_activation = activation_block()

    def build(self, input_shape):
        self.conv1 = self.conv_block(input_shape[-1], 3, 1)
        self.conv2 = self.conv_block(input_shape[-1], 3, 1)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv1(x, **kwargs)
        x = self.conv1_activation(x, **kwargs)
        x = self.conv2(x, **kwargs)
        x = self.sum_block([x, inputs])
        x = self.sum_activation(x, **kwargs)
        return x


class ResNetDownBlock(Model):
    kernel_size = 3

    def __init__(self, final_filters, stride,
                 conv_block: Model = ConvBn, activation_block: Model = ReLU, **kwargs):
        super().__init__(**kwargs)
        self.conv_block = conv_block
        self.stride = stride
        self.conv1 = None
        self.conv1_activation = activation_block()
        self.conv2 = None
        self.conv2_activation = activation_block()
        self.conv3 = conv_block(final_filters, 1, 1)

        self.projection = conv_block(final_filters, 1, stride)
        self.sum_block = tf.keras.layers.Add()
        self.sum_activation = activation_block()

    def build(self, input_shape):
        input_filters = input_shape[-1]
        bottleneck_filters = input_filters // 4
        # TODO probably using kernel 1 with stride 2 is not good, but so was in paper
        self.conv1 = self.conv_block(bottleneck_filters, 1, self.stride)
        self.conv2 = self.conv_block(input_filters, self.kernel_size, 1)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv1(x, **kwargs)
        x = self.conv1_activation(x, **kwargs)
        x = self.conv2(x, **kwargs)
        x = self.conv2_activation(x, **kwargs)
        x = self.conv3(x, **kwargs)
        x = self.sum_block([x, self.projection(inputs)])  # Add crashes with kwargs
        x = self.sum_activation(x, **kwargs)
        return x


class ClassificationHead(Sequential):
    def __init__(self, num_classes, **kwargs):
        super().__init__([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes)
        ], **kwargs)


class ResNetBackbone(Model):
    down_stride = 2

    def __init__(self, nf, num_repeats: List[int],
                 return_endpoints_on_call=False,
                 init_block: Model = InitialBlock,
                 main_block: Model = ResNetBottleneckBlock,
                 down_block: Model = ResNetDownBlock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_endpoints_on_call = return_endpoints_on_call

        self.blocks = [
            [
                init_block(nf)
            ]
        ]

        filters = nf * 4
        for i, n in enumerate(num_repeats):
            # note that first down block without stride
            block = [down_block(filters, self.down_stride if i > 0 else 1)]
            for _ in range(n):
                block.append(main_block())
            self.blocks.append(block)
            filters *= 2

    def call(self, inputs, **kwargs):
        endpoints = []
        x = inputs
        for block in self.blocks:
            for layer in block:
                x = layer(x, **kwargs)
            endpoints.append(x)
        if self.return_endpoints_on_call:
            return endpoints
        else:
            return endpoints[-1]


def get_resnet18_backbone(nf):
    return ResNetBackbone(nf, [1, 1, 1, 1], main_block=ResNetIdentityBlock)


def get_resnet34_backbone(nf):
    return ResNetBackbone(nf, [2, 3, 5, 2], main_block=ResNetIdentityBlock)


def get_resnet50_backbone(nf):
    return ResNetBackbone(nf, [2, 3, 5, 2])


def get_resnet101_backbone(nf):
    return ResNetBackbone(nf, [2, 3, 22, 2])


def get_resnet152_backbone(nf):
    return ResNetBackbone(nf, [2, 7, 35, 2])
