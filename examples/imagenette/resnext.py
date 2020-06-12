import os

from datasets.imagenette import get_data
from examples.imagenette.template import train
from models.architectures.resnext import get_resnext50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel

if __name__ == "__main__":
    nf = 64
    bs = 32
    name = "resnext50"

    resnext_backbone = get_resnext50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnext_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnext_backbone.save(backbone_export_path, include_optimizer=False)
