import os
import shutil

import tensorflow as tf
from unittest import TestCase

from datasets.imagenette import get_data
from examples.imagenette.template import train
from models.architectures.resnet_with_attention import get_se_resnet50_backbone, get_nl_resnet50_backbone, \
    get_ge_resnet50_backbone
from models.architectures.resnet import get_resnet50_backbone, get_resnet18_backbone
from models.architectures.resnext import get_resnext50_backbone
from models.architectures.xresnet import get_x_resnet50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel


def test_nn(name, backbone):
    bs = 64

    train_batches, validation_batches = get_data(bs)  # we use validation batches because we do not want augmentation
    train_batches = validation_batches.take(1)
    test_batches = train_batches

    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(backbone, head)

    export_path = os.path.join(name, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    head_export_path = os.path.join(export_path, "head")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train(model, name, train_batches, test_batches, 1, 0.1)
    loss1 = 0
    for test_batch in test_batches:
        embedding = backbone(test_batch[0])
        classes = head(embedding)
        loss1 = loss_fn(test_batch[1], classes)

    train(model, name, train_batches, test_batches, 20, 0.1)
    backbone.save(backbone_export_path, include_optimizer=False)
    head.save(head_export_path, include_optimizer=False)
    backbone = tf.keras.models.load_model(backbone_export_path, compile=False)
    head = tf.keras.models.load_model(head_export_path, compile=False)
    loss2 = 0
    loss2_2 = 0
    for test_batch in test_batches:
        # we use train=True because batch not moving averages have not trained enough yet and val loss very bad.
        embedding = backbone(test_batch[0], training=True)
        classes = head(embedding, training=True)
        loss2 = loss_fn(test_batch[1], classes)
        loss2_2 = loss_fn(test_batch[1], model(test_batch[0], training=True))
    assert tf.abs(loss2_2 - loss2) < 0.001  # checking save-load validity

    return loss2.numpy() < loss1.numpy()


class TestResNeXt50(TestCase):
    name = "resnext50_delme"
    nf = 16

    def test_resnext50(self):
        backbone = get_resnext50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestResNet50(TestCase):
    name = "resnet50_delme"
    nf = 16

    def test_resnet50(self):
        backbone = get_resnet50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestResNet18(TestCase):
    name = "resnet18_delme"
    nf = 16

    def test_resnet18(self):
        backbone = get_resnet18_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestXResNet50(TestCase):
    name = "xresnet50_delme"
    nf = 16

    def test_xresnet50(self):
        backbone = get_x_resnet50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestSEResNet50(TestCase):
    name = "se_resnet50_delme"
    nf = 16

    def test_se_resnet50(self):
        backbone = get_se_resnet50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestGEResNet50(TestCase):
    name = "ge_resnet50_delme"
    nf = 16

    def test_ge_resnet50(self):
        backbone = get_ge_resnet50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)


class TestNLResNet50(TestCase):
    name = "nl_resnet50_delme"
    nf = 16

    def test_nl_resnet50(self):
        backbone = get_nl_resnet50_backbone(self.nf)
        self.assertTrue(test_nn(self.name, backbone))

    def tearDown(self) -> None:
        if os.path.isdir(self.name):
            shutil.rmtree(self.name)
