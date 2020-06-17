import os

from datasets.imagenette import get_data
from examples.imagenette.template import train
from models.architectures.xresnet import get_x_resnet50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel

if __name__ == "__main__":
    nf = 64
    bs = 64
    name = "x_resnet50"

    xresnet_backbone = get_x_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)
