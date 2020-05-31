from typing import List

from tensorflow.python.keras import Model

from models.architectures.resnet import ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetIdentityDownBlockBuilder
from models.base_classes import ModelBuilder, SumBlockBuilder


class ResNeXtBlockBuilder(ModelBuilder):
    cardinality = 32
    bottleneck_filters = 4

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
        branches = [self.single_path_builder.build(
            filters=filters, stride=stride, bottleneck_filters=self.bottleneck_filters)
            for _ in range(self.cardinality)]
        branches_aggregation = self.aggregation_block_builder.build()
        return self.ResNeXtBlock(branches, branches_aggregation)


def get_resnext50_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_resnext101_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 22, 2], return_endpoints_on_call=False)


def get_resnext152_backbone(nf):
    main_resnet_block = ResNeXtBlockBuilder()
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetIdentityDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 7, 35, 2], return_endpoints_on_call=False)
