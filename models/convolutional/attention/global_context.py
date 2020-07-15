"""
GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
https://arxiv.org/pdf/1904.11492.pdf
2019
Implemented only final GC block, but not generalized global context framework.
At first I accidentally was using GC without softmax activation for keys. And that surprisingly showed better results.
So in some of my models I use linear activation instead of softmax, so be careful.
"""
import tensorflow as tf
from tensorflow.keras import Model

from models.convolutional.attention.non_local_network import ResNonLocalBlockBuilder, Conv1x1Builder
from models.base_classes import ModelBuilder, SumBlockBuilder, IdentityBlockBuilder, SoftmaxBuilder, ReLUBuilder, \
    LayerNormBuilder


class GCBlock(Model):
    def __init__(self, embedding_block_builder: ModelBuilder,
                 context_modeling_activation_block_builder: ModelBuilder,
                 transform_normalization_block_builder: ModelBuilder,
                 transform_activation_block_builder: ModelBuilder,
                 bottleneck_rate):
        super().__init__()
        self.embedding_block_builder = embedding_block_builder
        self.bottleneck_rate = bottleneck_rate

        self.keys_conv = embedding_block_builder.build(filters=1)
        self.keys_activation_block = context_modeling_activation_block_builder.build()

        self.v1_conv = None
        self.v1_norm = transform_normalization_block_builder.build()
        self.v1_activation = transform_activation_block_builder.build()
        self.v2_conv = None

    def build(self, input_shape):
        self.v1_conv = self.embedding_block_builder.build(filters=input_shape[-1] // self.bottleneck_rate)
        self.v2_conv = self.embedding_block_builder.build(filters=input_shape[-1])

    def call(self, inputs, training=None, mask=None):
        inputs_shape = inputs.get_shape()

        keys = self.keys_conv(inputs)  # shape [BS, H, W, 1]
        keys = tf.reshape(keys, [-1, inputs_shape[1]*inputs_shape[2], 1])  # shape [BS, HxW, 1]
        keys = tf.transpose(keys, perm=[0, 2, 1])  # shape [BS, 1, HxW]
        keys = self.keys_activation_block(keys)  # shape [BS, H, W, 1]
        values = inputs  # shape [BS, H, W, C]
        values = tf.reshape(values, [-1, inputs_shape[1]*inputs_shape[2], inputs_shape[-1]])  # shape [BS, HxW, C]

        result = tf.matmul(keys, values)  # shape [BS, 1, C]
        result = tf.reshape(result, [-1, 1, 1, inputs_shape[-1]])  # shape [BS, 1, 1, C]
        result = self.v1_conv(result)  # shape [BS, 1, 1, C/r]
        result = self.v1_norm(result)  # shape [BS, 1, 1, C/r]
        result = self.v1_activation(result)  # shape [BS, 1, 1, C/r]
        result = self.v2_conv(result)  # shape [BS, 1, 1, C]

        return result


class GCBlockBuilder(ModelBuilder):
    bottleneck_rate = 16

    def __init__(self, embedding_block_builder: ModelBuilder = Conv1x1Builder(),
                 context_modeling_activation_block_builder: ModelBuilder = SoftmaxBuilder(),
                 transform_normalization_block_builder: ModelBuilder = LayerNormBuilder(),
                 transform_activation_block_builder: ModelBuilder = ReLUBuilder()):
        self.embedding_block_builder = embedding_block_builder
        self.context_modeling_activation_block_builder = context_modeling_activation_block_builder
        self.transform_normalization_block_builder = transform_normalization_block_builder
        self.transform_activation_block_builder = transform_activation_block_builder

    def build(self, **kwargs) -> Model:
        return GCBlock(
            self.embedding_block_builder,
            self.context_modeling_activation_block_builder,
            self.transform_normalization_block_builder,
            self.transform_activation_block_builder,
            self.bottleneck_rate
        )


class ResGCBlockBuilder(ResNonLocalBlockBuilder):
    def __init__(self, gc_block_builder: ModelBuilder = GCBlockBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder(),
                 activation_block_builder: ModelBuilder = IdentityBlockBuilder()):
        super().__init__(
            non_local_block_builder=gc_block_builder,
            aggregation_block_builder=aggregation_block_builder, activation_block_builder=activation_block_builder)
