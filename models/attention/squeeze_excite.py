import tensorflow as tf
from tensorflow.keras import Model

from models.base_classes import ModelBuilder, ReLUBuilder, GlobalAvgPoolBuilder, \
    SigmoidBuilder, FCBlockBuilder


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
        def __init__(self, bottleneck_rate,
                     fc_block_builder: ModelBuilder,
                     global_pooling_block_builder: ModelBuilder,
                     internal_activation_builder: ModelBuilder,
                     output_activation_builder: ModelBuilder):
            super().__init__()
            self.bottleneck_rate = bottleneck_rate
            self.fc_block_builder = fc_block_builder
            self.global_pooling_block_builder = global_pooling_block_builder
            self.internal_activation_builder = internal_activation_builder
            self.output_activation_builder = output_activation_builder
            self.se_branch = None

        def build(self, input_shape):
            self.se_branch = tf.keras.Sequential([
                self.global_pooling_block_builder.build(),
                self.fc_block_builder.build(units=input_shape[-1] // self.bottleneck_rate),
                self.internal_activation_builder.build(),
                self.fc_block_builder.build(units=input_shape[-1]),
                self.output_activation_builder.build()
            ])

        def call(self, inputs, training=None, mask=None):
            channels_attention = self.se_branch(inputs, training=None, mask=None)
            channels_attention = tf.expand_dims(channels_attention, 1)
            channels_attention = tf.expand_dims(channels_attention, 1)
            result = inputs * channels_attention
            return result

    def build(self, **kwargs) -> Model:
        return self.SEBlock(
            self.bottleneck_rate,
            self.fc_block_builder,
            self.global_pooling_block_builder,
            self.internal_activation_builder,
            self.output_activation_builder
        )
