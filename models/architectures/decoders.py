from typing import List

import tensorflow as tf
from tensorflow.keras import Model

from models.architectures.resnet import ResNetBottleNeckBlockBuilder, ResNetProjectionDownBlockBuilder, \
    ResNetIdentityBlockBuilder, ConvBnBuilder
from models.architectures.xresnet import XResNetDBottleneckBlock, XResNetDProjectionBlock
from models.base_classes import ModelBuilder, ReLUBuilder, UpsampleBilinear, ConcatBlockBuilder


class UpsampleConvBnBuilder(ModelBuilder):
    def __init__(self, upsampling_block_builder: ModelBuilder = UpsampleBilinear()):
        super().__init__()
        self.upsampling_block_builder = upsampling_block_builder

    def build(self, filters, kernel_size=3, stride=1, **kwargs):
        seq = []
        if stride > 1:
            seq.append(self.upsampling_block_builder.build(stride=stride))
        seq.append(tf.keras.layers.Conv2D(filters, kernel_size, 1, use_bias=False, padding='same',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        seq.append(tf.keras.layers.BatchNormalization())
        return tf.keras.Sequential(seq, **kwargs)


class TransposedConvBnBuilder(ModelBuilder):
    def __init__(self):
        super().__init__()

    def build(self, filters, kernel_size=3, stride=1, **kwargs):
        seq = []
        if stride > 1:
            seq.append(tf.keras.layers.Conv2DTranspose(filters, kernel_size, stride, use_bias=False, padding="same",
                                                       kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        else:
            seq.append(tf.keras.layers.Conv2D(filters, kernel_size, 1, use_bias=False, padding='same',
                                              kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        seq.append(tf.keras.layers.BatchNormalization())
        return tf.keras.Sequential(seq, **kwargs)


class FinalConvBlockBuilder(ModelBuilder):
    kernel_size = 3
    pool_size = 3
    stride = 2

    def __init__(self, conv_block_builder: ModelBuilder = UpsampleConvBnBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder(),
                 upsampling_block_builder: ModelBuilder = UpsampleBilinear()):
        self.conv_block_builder = conv_block_builder
        self.activation_block_builder = activation_block_builder
        self.upsampling_block_builder = upsampling_block_builder

    def build(self, filters, output_filters, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.upsampling_block_builder.build(stride=self.stride),
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=output_filters, kernel_size=self.kernel_size, stride=self.stride)
        ], **kwargs)


class FinalConvBlockBuilder2(ModelBuilder):
    kernel_size = 4
    pool_size = 3
    stride = 2

    def __init__(self, conv_block_builder: ModelBuilder = ConvBnBuilder(),
                 activation_block_builder: ModelBuilder = ReLUBuilder(),
                 upsampling_block_builder: ModelBuilder = UpsampleConvBnBuilder()):
        self.conv_block_builder = conv_block_builder
        self.activation_block_builder = activation_block_builder
        self.upsampling_block_builder = upsampling_block_builder

    def build(self, filters, output_filters, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.upsampling_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=self.stride),
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.conv_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=1),
            self.activation_block_builder.build(),
            self.upsampling_block_builder.build(filters=filters, kernel_size=self.kernel_size, stride=self.stride),
            self.activation_block_builder.build(),
            tf.keras.layers.Conv2D(output_filters, 3, 1, 'same')  # TODO final pure conv (without BN)
        ], **kwargs)



class DecoderWithShortcutsBuilder(ModelBuilder):
    up_stride = 2

    def __init__(self, final_conv_builder: ModelBuilder = FinalConvBlockBuilder(),
                 main_block_builder: ModelBuilder = ResNetBottleNeckBlockBuilder(),
                 up_block_builder: ModelBuilder = ResNetProjectionDownBlockBuilder(),
                 shortcut_aggregation_block: ModelBuilder = ConcatBlockBuilder()):
        super().__init__()
        self.final_conv_builder = final_conv_builder
        self.main_block_builder = main_block_builder
        self.up_block_builder = up_block_builder
        self.shortcut_aggregation_block = shortcut_aggregation_block

    class DecoderModel(Model):
        def __init__(self, blocks: List[List[Model]]):
            super().__init__()
            self.blocks = blocks

        def call(self, inputs, training=None, mask=None):
            # inputs is a list of shortcuts (the last - the deepest)
            x = inputs[-1]
            for i, block in enumerate(self.blocks):
                if i > 0:
                    x = [x, inputs[-i - 1]]
                for layer in block:
                    x = layer(x, training=training, mask=mask)
            return x

    def build(self, filters, num_repeats: List[int], output_filters, **kwargs) -> Model:
        filters = filters * pow(2, len(num_repeats) + 1)

        blocks_sequence = []
        for i in range(len(num_repeats) - 1, -1, -1):
            n = num_repeats[i]

            block = []
            # note that first down block without stride
            if i < len(num_repeats) - 1:
                block.append(self.shortcut_aggregation_block.build())
            for _ in range(n):
                block.append(self.main_block_builder.build(filters=filters, stride=1))
            filters //= 2
            if i == 0:
                filters //= 2
            block.append(self.up_block_builder.build(filters=filters, stride=self.up_stride if i > 0 else 1))
            blocks_sequence.append(block)

        blocks_sequence.append([self.shortcut_aggregation_block.build(),
                                self.final_conv_builder.build(filters=filters, output_filters=output_filters)])
        return self.DecoderModel(blocks_sequence)


def get_x_resnet50_decoder(nf):
    return DecoderWithShortcutsBuilder(
        main_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder(
                conv_builder=UpsampleConvBnBuilder()
            )
        ),
        up_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=XResNetDBottleneckBlock(
                conv_block_builder=UpsampleConvBnBuilder()
            ),
            projection_block_builder=XResNetDProjectionBlock(
                conv_block_builder=UpsampleConvBnBuilder(),
                downsampling_block_builder=UpsampleBilinear()
            )
        )
    ).build(nf, [2, 3, 5, 2], 3)


def get_resnet18_decoder(nf):
    return DecoderWithShortcutsBuilder(
        main_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder(
                conv_builder=UpsampleConvBnBuilder()
            )
        ),
        up_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder(
                conv_builder=UpsampleConvBnBuilder()
            ),
            projection_block_builder=ResNetProjectionDownBlockBuilder(
                conv_block_builder=UpsampleConvBnBuilder(),
                projection_block_builder=UpsampleConvBnBuilder()
            )
        )
    ).build(nf, [1, 1, 1, 1], 3)


def get_resnet18_decoder_transposed(nf):
    return DecoderWithShortcutsBuilder(
        main_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder(
                conv_builder=TransposedConvBnBuilder()
            )
        ),
        up_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder(
                conv_builder=TransposedConvBnBuilder()
            ),
            projection_block_builder=ResNetProjectionDownBlockBuilder(
                conv_block_builder=TransposedConvBnBuilder(),
                projection_block_builder=TransposedConvBnBuilder()
            )
        ),
        final_conv_builder=FinalConvBlockBuilder2()
    ).build(nf, [1, 1, 1, 1], 3)
