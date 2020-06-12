from models.architectures.resnet import ConvBnBuilder, ResNetBlockBuilder
from models.base_classes import ModelBuilder, ReLUBuilder, MaxPoolBuilder, SumBlockBuilder, AvgPoolBuilder

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


class XResNetIdentityDownBlockBuilder(ModelBuilder):
    def __init__(self, conv_block_builder: ModelBuilder = ResNetBlockBuilder(),
                 projection_block_builder: ModelBuilder = ConvBnBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 downsampling_block_builder: ModelBuilder = AvgPoolBuilder(),
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
