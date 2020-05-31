from typing import List

import tensorflow as tf
from tensorflow.keras import Model

from models.base_classes import ModelBuilder, ReLUBuilder, SumBlockBuilder, MaxPoolBuilder


class ConvBnBuilder(ModelBuilder):
    def build(self, filters, kernel_size=3, stride=1, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, stride, use_bias=False, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization()
        ], **kwargs)


class InitialConvBlockBuilder(ModelBuilder):
    kernel_size = 7
    pool_size = 3
    stride = 2

    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder(),
                 downsampling_block_builder: ModelBuilder = MaxPoolBuilder()):
        self.conv_block_builder = conv_block_builder
        self.activation_block_builder = activation_block_builder
        self.downsampling_block_builder = downsampling_block_builder

    def build(self, filters, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=self.stride),
            self.activation_block_builder.build(),
            self.downsampling_block_builder.build()
        ], **kwargs)


class ResNetBlockBuilder(ModelBuilder):
    kernel_size = 3

    def __init__(self, conv_builder: ModelBuilder = ConvBnBuilder(), activation_builder: ModelBuilder = ReLUBuilder()):
        self.conv_builder = conv_builder
        self.activation_builder = activation_builder

    def build(self, filters, stride=1, **kwargs) -> Model:
        return tf.keras.Sequential([
            # TODO it is not very clear where to do stride, in first or second conv
            self.conv_builder.build(filters=filters, kernel_size=self.kernel_size, stride=stride),
            self.activation_builder.build(),
            self.conv_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
        ], **kwargs)


class ResNetIdentityBlockBuilder(ModelBuilder):
    def __init__(self, conv_block_builder: ModelBuilder = ResNetBlockBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder()):
        self.resnet_block_builder = conv_block_builder
        self.aggregation_block_builder = aggregation_block_builder
        self.activation_builder = activation_block_builder

    class ResNetIdentityBlock(Model):
        def __init__(self, resnet_block: Model, aggregation_block: Model, activation_block: Model, **kwargs):
            super().__init__(**kwargs)
            self.resnet_block = resnet_block
            self.aggregation_block = aggregation_block
            self.activation_block = activation_block

        def call(self, inputs, training=None, mask=None):
            x = inputs
            x = self.resnet_block(x, training=training, mask=mask)
            x = self.aggregation_block([x, inputs], training=training, mask=mask)
            x = self.activation_block(x, training=training, mask=mask)
            return x

    def build(self, filters, stride, **kwargs) -> Model:
        return self.ResNetIdentityBlock(
            resnet_block=self.resnet_block_builder.build(filters=filters, stride=stride),
            aggregation_block=self.aggregation_block_builder.build(),
            activation_block=self.activation_builder.build()
        )


class ResNetIdentityDownBlockBuilder(ModelBuilder):
    def __init__(self, conv_block_builder: ModelBuilder = ResNetBlockBuilder(),
                 projection_block_builder: ModelBuilder = ConvBnBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder()):
        self.conv_block_builder = conv_block_builder
        self.projection_block_builder = projection_block_builder
        self.aggregation_block_builder = aggregation_block_builder
        self.activation_builder = activation_block_builder

    class ResNetIdentityBlock(Model):
        def __init__(self, resnet_block: Model, projection_block: Model,
                     aggregation_block: Model, activation_block: Model, **kwargs):
            super().__init__(**kwargs)
            self.resnet_block = resnet_block
            self.projection_block = projection_block
            self.aggregation_block = aggregation_block
            self.activation_block = activation_block

        def call(self, inputs, training=None, mask=None):
            x = inputs
            x = self.resnet_block(x, training=training, mask=mask)
            projection = self.projection_block(inputs, training=training, mask=mask)
            x = self.aggregation_block([x, projection], training=training, mask=mask)
            x = self.activation_block(x, training=training, mask=mask)
            return x

    def build(self, filters, stride, **kwargs) -> Model:
        return self.ResNetIdentityBlock(
            resnet_block=self.conv_block_builder.build(filters=filters, stride=stride),
            projection_block=self.projection_block_builder.build(filters=filters, kernel_size=1, stride=stride),
            aggregation_block=self.aggregation_block_builder.build(),
            activation_block=self.activation_builder.build()
        )


class ResNetBottleNeckBlockBuilder(ModelBuilder):
    kernel_size = 3

    def __init__(self, conv_builder: ModelBuilder = ConvBnBuilder(), activation_builder: ModelBuilder = ReLUBuilder()):
        self.conv_builder = conv_builder
        self.activation_builder = activation_builder

    def build(self, filters, stride=1, bottleneck_filters=None, **kwargs) -> Model:
        bottleneck_filters = bottleneck_filters if bottleneck_filters in kwargs else filters // 4
        return tf.keras.Sequential([
            self.conv_builder.build(filters=bottleneck_filters, kernel_size=1, stride=stride),
            self.activation_builder.build(),
            self.conv_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_builder.build(),
            self.conv_builder.build(filters=filters, kernel_size=1, stride=1)
        ], **kwargs)


class ResNetBackboneBuilder(ModelBuilder):
    down_stride = 2

    def __init__(self, init_conv_builder: ModelBuilder = InitialConvBlockBuilder(),
                 resnet_block_builder: ModelBuilder = ResNetIdentityBlockBuilder(),
                 resnet_down_block_builder: ModelBuilder = ResNetIdentityDownBlockBuilder()):
        self.init_conv_builder = init_conv_builder
        self.resnet_block_builder = resnet_block_builder
        self.resnet_down_block_builder = resnet_down_block_builder

    class ResNetBackbone(Model):
        def __init__(self, blocks: List[List[Model]], return_endpoints_on_call=False, **kwargs):
            super().__init__(**kwargs)
            self.blocks = blocks
            self.return_endpoints_on_call = return_endpoints_on_call

        def call(self, inputs, training=None, mask=None):
            endpoints = []
            x = inputs
            for block in self.blocks:
                for layer in block:
                    x = layer(x, training=training, mask=mask)
                endpoints.append(x)
            if self.return_endpoints_on_call:
                return endpoints
            else:
                return endpoints[-1]

    def build(self, filters, num_repeats: List[int], return_endpoints_on_call=False, **kwargs) -> Model:
        blocks_sequence = [
            [self.init_conv_builder.build(filters=filters)]
        ]
        filters *= 4
        for i, n in enumerate(num_repeats):
            # note that first down block without stride
            block = [self.resnet_down_block_builder.build(filters=filters,
                                                          stride=self.down_stride if i > 0 else 1)]
            for _ in range(n):
                block.append(self.resnet_block_builder.build(filters=filters, stride=1))
            blocks_sequence.append(block)
            filters *= 2
        return self.ResNetBackbone(blocks_sequence, return_endpoints_on_call=return_endpoints_on_call)


def get_resnet18_backbone(nf):
    return ResNetBackboneBuilder().build(nf, [1, 1, 1, 1], return_endpoints_on_call=False)


def get_resnet34_backbone(nf):
    return ResNetBackboneBuilder().build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_resnet50_backbone(nf):
    main_resnet_block = ResNetBottleNeckBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_resnet101_backbone(nf):
    main_resnet_block = ResNetBottleNeckBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 22, 2], return_endpoints_on_call=False)


def get_resnet152_backbone(nf):
    main_resnet_block = ResNetBottleNeckBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 7, 35, 2], return_endpoints_on_call=False)
