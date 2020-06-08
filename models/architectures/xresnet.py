from models.architectures.resnet import ConvBnBuilder, ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder
from models.base_classes import ModelBuilder, ReLUBuilder, MaxPoolBuilder

import tensorflow as tf
from tensorflow.keras import Model


class XResNetInitialConvBlockBuilder(ModelBuilder):
    kernel_size = 3
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
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.downsampling_block_builder.build()
        ], **kwargs)


class XResNetDBottleneckBlock(ModelBuilder):
    kernel_size = 3
    stride = 2

    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder()):
        self.conv_block_builder = conv_block_builder
        self.activation_block_builder = activation_block_builder

    def build(self, filters, stride, bottleneck_filters=None, **kwargs) -> Model:
        bottleneck_filters = bottleneck_filters if bottleneck_filters else filters // 4
        return tf.keras.Sequential([
            self.conv_block_builder.build(filters=bottleneck_filters, kernel_size=1, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=bottleneck_filters, kernel_size=self.kernel_size, stride=self.stride),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=1, stride=1)
        ], **kwargs)


class XResNetDProjectionBlock(ModelBuilder):
    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 downsampling_block_builder: ModelBuilder = MaxPoolBuilder()):
        self.conv_block_builder = conv_block_builder
        self.downsampling_block_builder = downsampling_block_builder

    def build(self, filters, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.downsampling_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=1, stride=1)
        ], **kwargs)


def get_xresnet50_backbone(nf):
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder()
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=XResNetDBottleneckBlock(),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)
