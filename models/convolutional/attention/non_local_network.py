"""
https://arxiv.org/abs/1711.07971
Implemented spatial-only embedded Gaussian version with subsampling trick (with max-pooling).
Applying it to all res blocks leads to some instability (probably exploding gradients).
But using 5 blocks (2 in res3 and 3 in res4) works fine.
"""
from typing import Tuple, List

from models.architectures.resnet import ResNetIdentityBlockBuilder
from models.attention.base import BlockWithPostAttentionBuilder
from models.base_classes import ModelBuilder, MaxPoolBuilder, SoftmaxBuilder, \
    SumBlockBuilder, IdentityBlockBuilder

import tensorflow as tf
from tensorflow.keras import Model


class Conv1x1Builder(ModelBuilder):
    def build(self, filters, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, 1, use_bias=False, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))
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
        self.keys_pooling_block = pooling_block_builder.build()
        self.values_pooling_block = pooling_block_builder.build()
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
        inputs_shape = inputs.get_shape()

        query = self.query_conv(inputs)
        keys = self.keys_conv(inputs)
        values = self.values_conv(inputs)

        # not that we do not pool query, so inputs and attention tensors will match
        keys = self.keys_pooling_block(keys)
        values = self.values_pooling_block(values)

        pooled_shape = keys.get_shape()

        query = tf.reshape(query, [-1, inputs_shape[1]*inputs_shape[2], pooled_shape[3]])
        keys = tf.reshape(keys, [-1, pooled_shape[1]*pooled_shape[2], pooled_shape[3]])
        values = tf.reshape(values, [-1, pooled_shape[1]*pooled_shape[2], pooled_shape[3]])
        attention = tf.linalg.matmul(query, keys, transpose_b=True)
        attention = self.activation_block(attention)

        result = tf.linalg.matmul(attention, values)
        result = tf.reshape(result, [-1, inputs_shape[1], inputs_shape[2], pooled_shape[3]])
        result = self.post_attention_conv(result)

        return result


class NonLocalBlockBuilder(ModelBuilder):
    bottleneck_rate = 2

    def __init__(self, embedding_block_builder: ModelBuilder = Conv1x1Builder(),
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
    def __init__(self, non_local_block_builder: ModelBuilder = NonLocalBlockBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 activation_block_builder: ModelBuilder = IdentityBlockBuilder()):
        super().__init__(
            conv_block_builder=non_local_block_builder,
            aggregation_block_builder=aggregation_block_builder, activation_block_builder=activation_block_builder)


class AttentionInSpecifiedResNetLocations(BlockWithPostAttentionBuilder):
    def __init__(self, locations: List[Tuple], main_block_builder: ModelBuilder, attention_block_builder: ModelBuilder):
        super().__init__(main_block_builder, attention_block_builder)
        self.locations = locations

    def build(self, filters, stride=1, n_stage=-1, n_block=-1, **kwargs) -> tf.keras.Model:
        if (n_stage, n_block) in self.locations:
            return super().build(filters=filters, stride=stride, **kwargs)
        else:
            return self.main_block_builder.build(filters=filters, stride=stride, **kwargs)
