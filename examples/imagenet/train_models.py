import os

from datasets.imagenet import get_data
from examples.imagenet.template import train
from models.architectures.resnet import get_resnet50_backbone, get_resnet18_backbone, \
    get_resnet18_with_bottleneck_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel


def train_resnet18():  # TODO not trained
    nf = 32
    bs = 112
    name = "resnet18"

    resnet_backbone = get_resnet18_with_bottleneck_backbone(nf)
    head = ClassificationHeadBuilder().build(1000)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    head_export_path = os.path.join(export_path, "head")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)
    head.save(head_export_path, include_optimizer=False)


def train_resnet50():  # TODO not trained
    nf = 64
    bs = 32
    name = "resnet50"

    resnet_backbone = get_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(1000)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


train_resnet18()
