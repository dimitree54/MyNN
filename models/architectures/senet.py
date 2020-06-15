from models.architectures.resnet import ConvBnBuilder, ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder
from models.architectures.resnext import ResNeXtBlockBuilderB
from models.base_classes import ModelBuilder, ReLUBuilder, MaxPoolBuilder, AvgPoolBuilder, GlobalAvgPoolBuilder, \
    SigmoidBuilder, FCBlockBuilder

import tensorflow as tf
from tensorflow.keras import Model


class SEBlockBuilder(ModelBuilder):
    bottleneck_rate = 16

    def __init__(self, fc_block_builder: ModelBuilder = FCBlockBuilder(),
                 global_pooling_block_builder: ModelBuilder = GlobalAvgPoolBuilder(),
                 internal_activation_builder: ModelBuilder = ReLUBuilder(),
                 output_activation_builder: ModelBuilder = SigmoidBuilder()):
        self.fc_block_builder = fc_block_builder
        self.global_pooling_block_builder = global_pooling_block_builder
        self.internal_activation_builder = internal_activation_builder
        self.output_activation_builder = output_activation_builder

    class SEBlock(Model):
        def __init__(self):
            super().__init__()
            self.se_branch = None

        def build(self, input_shape):
            self.se_branch = tf.keras.Sequential([
                self.global_pooling_block_builder.build(),
                self.fc_block_builder.build(input_shape[-1] // self.bottleneck_rate),
                self.internal_activation_builder.build(),
                self.fc_block_builder.build(input_shape[-1]),
                self.output_activation_builder.build()
            ])

        def call(self, inputs, training=None, mask=None):
            channels_attention = self.se_branch(inputs, training=None, mask=None)
            channels_attention = tf.expand_dims(channels_attention, 1)
            channels_attention = tf.expand_dims(channels_attention, 1)
            result = inputs * channels_attention
            return result

    def build(self, filters, **kwargs) -> Model:
        return self.SEBlock()


class BlockWithSEBuilder(ModelBuilder):
    def __init__(self, main_block_builder: ModelBuilder = ConvBnBuilder(),
                 se_block_builder: ModelBuilder = SEBlockBuilder()):
        self.main_block_builder = main_block_builder
        self.se_block_builder = se_block_builder

    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.main_block_builder.build(**kwargs),
            self.se_block_builder.build(**kwargs)
        ], **kwargs)


class XResNetDBottleneckBlock(ModelBuilder):
    kernel_size = 3

    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder()):
        self.conv_block_builder = conv_block_builder
        self.activation_block_builder = activation_block_builder

    def build(self, filters, stride, bottleneck_filters=None, **kwargs) -> Model:
        bottleneck_filters = bottleneck_filters if bottleneck_filters else filters // 4
        return tf.keras.Sequential([
            self.conv_block_builder.build(filters=bottleneck_filters, kernel_size=1, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=bottleneck_filters, kernel_size=self.kernel_size, stride=stride),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=1, stride=1)
        ], **kwargs)


class XResNetDProjectionBlock(ModelBuilder):
    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 downsampling_block_builder: ModelBuilder = MaxPoolBuilder()):
        self.conv_block_builder = conv_block_builder
        self.downsampling_block_builder = downsampling_block_builder

    def build(self, filters, kernel_size=1, stride=2, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.downsampling_block_builder.build(stride=stride),
            self.conv_block_builder.build(filters=filters, kernel_size=kernel_size, stride=1)
        ], **kwargs)


class XResNeXtBlockBuilderB(ResNeXtBlockBuilderB):
    def build_branch(self, bottleneck_filters, stride, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.conv_builder.build(filters=bottleneck_filters, kernel_size=1, stride=1, **kwargs),
            self.conv_builder.build(filters=bottleneck_filters, kernel_size=3, stride=stride, **kwargs)
        ])
