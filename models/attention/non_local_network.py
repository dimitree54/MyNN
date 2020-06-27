"""
https://arxiv.org/abs/1711.07971
Implemented spatial-only embedded Gaussian version with subsampling trick (with max-pooling).
TODO not tested
"""
from models.architectures.resnet import ResNetIdentityBlockBuilder
from models.base_classes import ModelBuilder, MaxPoolBuilder, SoftmaxBuilder, \
    SumBlockBuilder, IdentityBlockBuilder

import tensorflow as tf
from tensorflow.keras import Model


class Conv1x1BnBuilder(ModelBuilder):
    def build(self, filters, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, 1, use_bias=False, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization()
        ], **kwargs)


class NonLocalBlock(Model):
    def __init__(self, embedding_block_builder: ModelBuilder,
                 pooling_block_builder: ModelBuilder,
                 activation_block_builder: ModelBuilder,
                 bottleneck_rate=2):
        super().__init__()
        self.embedding_block_builder = embedding_block_builder
        self.pooling_block_builder = pooling_block_builder
        self.activation_block_builder = activation_block_builder
        self.bottleneck_rate = bottleneck_rate

        self.query_conv = None
        self.keys_conv = None
        self.values_conv = None
        self.query_pooling_block = pooling_block_builder.build()
        self.keys_pooling_block = pooling_block_builder.build()
        self.activation_block = activation_block_builder.build()
        self.post_attention_conv = None
        self.bottleneck_channels = 0

    def build(self, input_shape):
        self.bottleneck_channels = input_shape[-1]//self.bottleneck_rate
        self.query_conv = self.embedding_block_builder.build(filters=self.bottleneck_channels)
        self.keys_conv = self.embedding_block_builder.build(filters=self.bottleneck_channels)
        self.values_conv = self.embedding_block_builder.build(filters=self.bottleneck_channels)
        self.post_attention_conv = self.embedding_block_builder.build(filters=input_shape[-1])

    def call(self, inputs, training=None, mask=None):
        inputs_shape = tf.shape(inputs)

        query = self.keys_conv(inputs)
        keys = self.keys_conv(inputs)
        values = self.keys_conv(inputs)

        query = self.query_pooling_block(query)
        keys = self.keys_pooling_block(keys)

        query = tf.reshape(query, [inputs_shape[0], inputs_shape[1]*inputs_shape[2], inputs_shape[3]])
        keys = tf.reshape(keys, [inputs_shape[0], inputs_shape[1]*inputs_shape[2], inputs_shape[3]])
        values = tf.reshape(values, [inputs_shape[0], inputs_shape[1]*inputs_shape[2], inputs_shape[3]])

        attention = tf.linalg.matmul(query, keys, transpose_b=True)
        attention = self.activation_block(attention)

        result = tf.linalg.matmul(attention, values)
        result = self.post_attention_conv(result)

        return result


class NonLocalBlockBuilder(ModelBuilder):
    bottleneck_rate = 2

    def __init__(self, embedding_block_builder: ModelBuilder = Conv1x1BnBuilder(),  # TODO make this Conv always 1x1
                 pooling_block_builder: ModelBuilder = MaxPoolBuilder(),
                 activation_block_builder: ModelBuilder = SoftmaxBuilder()):
        self.embedding_block_builder = embedding_block_builder
        self.pooling_block_builder = pooling_block_builder
        self.activation_block_builder = activation_block_builder

    def build(self, **kwargs) -> Model:
        return NonLocalBlock(
            self.embedding_block_builder,
            self.pooling_block_builder,
            self.activation_block_builder,
            self.bottleneck_rate
        )


class ResNonLocalBlockBuilder(ResNetIdentityBlockBuilder):
    def __init__(self, embedding_block_builder: ModelBuilder = Conv1x1BnBuilder(),  # TODO make this Conv always 1x1
                 pooling_block_builder: ModelBuilder = MaxPoolBuilder(),
                 attention_activation_block_builder: ModelBuilder = SoftmaxBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 activation_block_builder: ModelBuilder = IdentityBlockBuilder()):
        super().__init__(
            conv_block_builder=NonLocalBlockBuilder(embedding_block_builder, pooling_block_builder,
                                                    attention_activation_block_builder),
            aggregation_block_builder=aggregation_block_builder, activation_block_builder=activation_block_builder)
