import tensorflow as tf
from tensorflow.keras import Model

from models.architectures.resnet import ConvBnBuilder, ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder
from models.architectures.xresnet import XResNeXtBlockBuilderB, XResNetInitialConvBlockBuilder, \
    XResNetDProjectionBlock, XResNetDBottleneckBlock
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
                     fc_block_builder: ModelBuilder = FCBlockBuilder(),
                     global_pooling_block_builder: ModelBuilder = GlobalAvgPoolBuilder(),
                     internal_activation_builder: ModelBuilder = ReLUBuilder(),
                     output_activation_builder: ModelBuilder = SigmoidBuilder()):
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


class BlockWithSEBuilder(ModelBuilder):
    def __init__(self, main_block_builder: ModelBuilder = ConvBnBuilder(),
                 se_block_builder: ModelBuilder = SEBlockBuilder()):
        self.main_block_builder = main_block_builder
        self.se_block_builder = se_block_builder

    def build(self, filters, stride=1, **kwargs) -> Model:
        return tf.keras.Sequential([
            self.main_block_builder.build(filters=filters, stride=stride, **kwargs),
            self.se_block_builder.build(**kwargs)
        ], **kwargs)


def get_se_resnet50_backbone(nf):
    main_resnet_block = BlockWithSEBuilder(main_block_builder=ResNetBottleNeckBlockBuilder())
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xse_resnet50_backbone(nf):
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=BlockWithSEBuilder(ResNetBottleNeckBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=BlockWithSEBuilder(XResNetDBottleneckBlock()),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xse_resnext50_backbone(nf):
    main_resnet_block = BlockWithSEBuilder(XResNeXtBlockBuilderB())
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block,
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)
