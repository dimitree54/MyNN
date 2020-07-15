from models.base_classes import IdentityBlockBuilder
from models.convolutional.attention.global_context import ResGCBlockBuilder, GCBlockBuilder
from models.convolutional.architectures.resnet import ResNetBottleNeckBlockBuilder, ResNetBackboneBuilder, \
    ResNetIdentityBlockBuilder, ResNetProjectionDownBlockBuilder
from models.convolutional.architectures.xresnet import XResNetInitialConvBlockBuilder, XResNeXtBlockBuilderB, \
    XResNetDProjectionBlock, XResNetDBottleneckBlock
from models.convolutional.attention.base import BlockWithPostAttentionBuilder
from models.convolutional.attention.gather_excite import GEBlockBuilder
from models.convolutional.attention.non_local_network import ResNonLocalBlockBuilder, \
    AttentionInSpecifiedResNetLocations
from models.convolutional.attention.squeeze_excite import SEBlockBuilder


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


def get_xge_resnext50_backbone(nf):
    main_resnet_block = BlockWithPostAttentionBuilder(
        main_block_builder=XResNeXtBlockBuilderB(), attention_block_builder=GEBlockBuilder())
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


def get_nl_resnet50_backbone(nf):  # non-local attention
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=AttentionInSpecifiedResNetLocations(
                locations=[(1, 0), (1, 2), (2, 0), (2, 2), (2, 4)],
                # best config from paper: 2 blocks in res3 and 3 in res5
                main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=ResNonLocalBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xnl_resnet50_backbone(nf):
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=AttentionInSpecifiedResNetLocations(
                locations=[(1, 0), (1, 2), (2, 0), (2, 2), (2, 4)],
                main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=ResNonLocalBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=XResNetDBottleneckBlock(),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_gc_resnet50_backbone(nf):  # non-local attention
    return ResNetBackboneBuilder(
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=ResNetBottleNeckBlockBuilder(), attention_block_builder=ResGCBlockBuilder())
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=ResNetBottleNeckBlockBuilder()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)


def get_xgc_resnet50_backbone(nf):
    # WARNING, I use linear activation for GC here instead of softmax.
    return ResNetBackboneBuilder(
        init_conv_builder=XResNetInitialConvBlockBuilder(),
        resnet_block_builder=ResNetIdentityBlockBuilder(
            conv_block_builder=BlockWithPostAttentionBuilder(
                main_block_builder=ResNetBottleNeckBlockBuilder(),
                attention_block_builder=ResGCBlockBuilder(
                    gc_block_builder=GCBlockBuilder(
                        context_modeling_activation_block_builder=IdentityBlockBuilder()  # linear activation
                    )
                )
            )
        ),
        resnet_down_block_builder=ResNetProjectionDownBlockBuilder(
            conv_block_builder=XResNetDBottleneckBlock(),
            projection_block_builder=XResNetDProjectionBlock()
        )
    ).build(nf, [2, 3, 5, 2], return_endpoints_on_call=False)
