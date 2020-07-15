import os

from datasets.imagenette import get_data
from examples.imagenette.template import train
from models.architectures.resnet import get_resnet50_backbone
from models.architectures.resnext import get_resnext50_backbone
from models.architectures.resnet_with_attention import get_ge_resnet50_backbone, get_se_resnet50_backbone, \
    get_xse_resnext50_backbone, \
    get_xse_resnet50_backbone, get_xge_resnet50_backbone, get_xge_resnext50_backbone, get_nl_resnet50_backbone, \
    get_xnl_resnet50_backbone, get_gc_resnet50_backbone, get_xgc_resnet50_backbone
from models.architectures.xresnet import get_x_resnet50_backbone, get_x_resnext50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel


def train_resnet50():
    nf = 64
    bs = 64
    name = "resnet50"

    resnet_backbone = get_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_resnext50():
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


def train_se_resnet50():
    nf = 64
    bs = 32
    name = "se_resnet50"

    resnet_backbone = get_se_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_ge_resnet50():
    nf = 64
    bs = 32
    name = "ge_resnet50"

    resnet_backbone = get_ge_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_x_resnet50():
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


def train_x_resnext50():
    nf = 64
    bs = 32
    name = "x_resnext50"

    xresnet_backbone = get_x_resnext50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xse_resnet50():
    nf = 64
    bs = 32
    name = "xse_resnet50"

    xresnet_backbone = get_xse_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xse_resnext50():
    nf = 64
    bs = 32
    name = "xse_resnext50"

    xresnet_backbone = get_xse_resnext50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xge_resnet50():
    nf = 64
    bs = 32
    name = "xge_resnet50"

    xresnet_backbone = get_xge_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xge_resnext50():
    nf = 64
    bs = 32
    name = "xge_resnext50"

    xresnet_backbone = get_xge_resnext50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_nl_resnet50():
    nf = 64
    bs = 32
    name = "nl_resnet50"

    resnet_backbone = get_nl_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xnl_resnet50():
    nf = 64
    bs = 32
    name = "xnl_resnet50"

    xresnet_backbone = get_xnl_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_gc_resnet50():
    nf = 64
    bs = 32
    name = "gc_resnet50"

    resnet_backbone = get_gc_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(resnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    resnet_backbone.save(backbone_export_path, include_optimizer=False)


def train_xgc_resnet50():
    nf = 64
    bs = 32
    name = "xgc_resnet50"

    xresnet_backbone = get_xgc_resnet50_backbone(nf)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(xresnet_backbone, head)

    train_batches, validation_batches = get_data(bs)

    train(model, name, train_batches, validation_batches)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    xresnet_backbone.save(backbone_export_path, include_optimizer=False)
