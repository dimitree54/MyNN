import os

from datasets.imagenet import get_data
from examples.imagenet.template import train
from models.architectures.resnet import get_resnet50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel

if __name__ == "__main__":
    nf = 64
    bs = 32
    name = "resnet50"

    resnet_backbone = get_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(resnet_backbone, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)
