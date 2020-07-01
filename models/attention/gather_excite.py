"""
https://arxiv.org/pdf/1810.12348.pdf

Here implemented only block with best performance (but most computationally expensive) GEBlockThetaPlus
"""
from models.architectures.resnet import ConvBnBuilder
from models.base_classes import ModelBuilder, ReLUBuilder, SigmoidBuilder

import tensorflow as tf
from tensorflow.keras import Model


class DepthwiseConvBnBuilder(ModelBuilder):
    def build(self, filters, kernel_size=3, stride=1, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size, stride, use_bias=False, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization()
        ], **kwargs)


class GEBlockThetaPlus(Model):
    def __init__(self, init_conv_block_builder: ModelBuilder,
                 internal_conv_block_builder: ModelBuilder,
                 internal_activation_builder: ModelBuilder,
                 output_activation_builder: ModelBuilder):
        super().__init__()
        self.init_conv_block_builder = init_conv_block_builder
        self.internal_conv_block_builder = internal_conv_block_builder
        self.internal_activation_builder = internal_activation_builder
        self.output_activation_builder = output_activation_builder
        self.ge_branch = None

    def build(self, input_shape):
        self.ge_branch = tf.keras.Sequential([
            self.init_conv_block_builder.build(filters=input_shape[-1],
                                               kernel_size=input_shape[1:3], stride=input_shape[1:3]),
            self.internal_activation_builder.build(),
            self.internal_conv_block_builder.build(filters=input_shape[-1], kernel_size=1),
            self.output_activation_builder.build()
        ])

    def call(self, inputs, training=None, mask=None):
        channels_attention = self.ge_branch(inputs, training=None, mask=None)
        result = inputs * channels_attention
        return result


class GEBlockBuilder(ModelBuilder):
    def __init__(self,
                 init_conv_block_builder: ModelBuilder = DepthwiseConvBnBuilder(),
                 internal_conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 internal_activation_builder: ModelBuilder = ReLUBuilder(),
                 output_activation_builder: ModelBuilder = SigmoidBuilder()):
        self.init_conv_block_builder = init_conv_block_builder
        self.internal_conv_block_builder = internal_conv_block_builder
        self.internal_activation_builder = internal_activation_builder
        self.output_activation_builder = output_activation_builder

    def build(self, **kwargs) -> Model:
        return GEBlockThetaPlus(
            self.init_conv_block_builder,
            self.internal_conv_block_builder,
            self.internal_activation_builder,
            self.output_activation_builder
        )
