from typing import List

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential

from models.convolutional.architectures.resnet import ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder, ConvBnBuilder
from models.base_classes import ModelBuilder, SumBlockBuilder


class ResNeXtBlockBuilderA(ModelBuilder):
    """
    This implementation does not work it takes more memory abd increases loss
    """
    cardinality = 32
    base_bottleneck_filters = 4

    def __init__(self,
                 single_path_builder: ModelBuilder = ResNetBottleNeckBlockBuilder(),
                 aggregation_block_builder: ModelBuilder = SumBlockBuilder()
                 ):
        self.single_path_builder = single_path_builder
        self.aggregation_block_builder = aggregation_block_builder

    class ResNeXtBlock(Model):
        def __init__(self, branches: List[Model], branches_aggregation: Model, **kwargs):
            super().__init__(**kwargs)
            self.branches = branches
            self.branches_aggregation = branches_aggregation

        def call(self, inputs, training=None, mask=None):
            x = [branch(inputs, training=training, mask=mask) for branch in self.branches]
            x = self.branches_aggregation(x, training=training, mask=mask)
            return x

    def build(self, filters, stride=1, **kwargs) -> Model:
        # for layer with 256 input filters we have base_bottleneck_filters as bottleneck size and we increase it
        # this bottleneck size proportionally to input filters.
        bottleneck_filters = self.base_bottleneck_filters * filters // 4 // 64
        branches = [self.single_path_builder.build(
            filters=filters, stride=stride, bottleneck_filters=bottleneck_filters)
            for _ in range(self.cardinality)]
        branches_aggregation = self.aggregation_block_builder.build()
        return self.ResNeXtBlock(branches, branches_aggregation)


class ResNeXtBlockBuilderB(ModelBuilder):
    cardinality = 32
    base_bottleneck_filters = 4

    def __init__(self,
                 conv_builder: ModelBuilder = ConvBnBuilder()):
        self.conv_builder = conv_builder

    class ResNeXtBlock(Model):
        def __init__(self, branches: List[Model], final_conv: Model, **kwargs):
            super().__init__(**kwargs)
            self.branches = branches
            self.branches_aggregation = tf.keras.layers.Concatenate()
            self.final_conv = final_conv

        def call(self, inputs, training=None, mask=None):
            x = [branch(inputs, training=training, mask=mask) for branch in self.branches]
            x = self.branches_aggregation(x)
            x = self.final_conv(x, training=training, mask=mask)
            return x

    def build_branch(self, bottleneck_filters, stride, **kwargs) -> Model:
        return Sequential([
            self.conv_builder.build(filters=bottleneck_filters, kernel_size=1, stride=stride, **kwargs),
            self.conv_builder.build(filters=bottleneck_filters, kernel_size=3, stride=1, **kwargs)
        ])

    def build(self, filters, stride=1, **kwargs) -> Model:
        # for layer with 256 input filters we have base_bottleneck_filters as bottleneck size and we increase it
        # this bottleneck size proportionally to input filters.
        bottleneck_filters = self.base_bottleneck_filters * filters // 4 // 64
        branches = [self.build_branch(bottleneck_filters, stride, **kwargs)
                    for _ in range(self.cardinality)]
        final_conv = self.conv_builder.build(filters=filters, kernel_size=1, stride=1)
        return self.ResNeXtBlock(branches, final_conv)


def get_resnext50_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilderB()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_resnext101_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilderB()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 22, 2], return_endpoints_on_call=False)


def get_resnext152_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilderB()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 7, 35, 2], return_endpoints_on_call=False)
