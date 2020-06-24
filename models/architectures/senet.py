from models.architectures.resnet import ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder
from models.architectures.xresnet import XResNetInitialConvBlockBuilder, XResNeXtBlockBuilderB, \
    XResNetDProjectionBlock, XResNetDBottleneckBlock
from models.attention.base import BlockWithPostAttentionBuilder
from models.attention.gather_excite import GEBlockBuilder
from models.attention.squeeze_excite import SEBlockBuilder


def get_se_resnet50_backbone(nf):
    main_resnet_block = BlockWithPostAttentionBuilder(
        main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=SEBlockBuilder())
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=main_resnet_block
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=main_resnet_block
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_ge_resnet50_backbone(nf):
    main_resnet_block = BlockWithPostAttentionBuilder(
        main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=GEBlockBuilder())
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
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=SEBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=XResNetDBottleneckBlock(), attention_block_builder=SEBlockBuilder()),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xge_resnet50_backbone(nf):
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=GEBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=XResNetDBottleneckBlock(), attention_block_builder=GEBlockBuilder()),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xse_resnext50_backbone(nf):
    main_resnet_block = BlockWithPostAttentionBuilder(
        main_block_builder=XResNeXtBlockBuilderB(), attention_block_builder=SEBlockBuilder())
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
