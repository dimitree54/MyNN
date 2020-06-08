import os
import shutil

import tensorflow as tf
from unittest import TestCase

from datasets.imagenette import get_data
from examples.imagenette.template import main
from models.architectures.resnet import get_resnet50_backbone, get_resnet18_backbone
from models.architectures.resnext import get_resnext50_backbone
from models.architectures.xresnet import get_xresnet50_backbone


class TestResNeXt(TestCase):
    def test_resnext50(self):
        nf = 16
        bs = 4
        resnext_backbone = get_resnext50_backbone(nf)

        train_batches, _ = get_data(bs)
        train_batches = train_batches.take(10)
        test_batches = train_batches.take(1)

        name = "resnext50_delme"
        export_path = os.path.join(name, "export")
        backbone_export_path = os.path.join(export_path, "backbone")
        head_export_path = os.path.join(export_path, "head")

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        main(resnext_backbone, name, train_batches, test_batches, 1)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss1 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss1 = loss_fn(test_batch[1], classes)

        main(resnext_backbone, name, train_batches, test_batches, 5)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss2 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss2 = loss_fn(test_batch[1], classes)

        shutil.rmtree(name)
        self.assertLess(loss2, loss1)


class TestResNet(TestCase):
    def test_resnet50(self):
        nf = 16
        bs = 4
        resnext_backbone = get_resnet50_backbone(nf)

        train_batches, _ = get_data(bs)
        train_batches = train_batches.take(10)
        test_batches = train_batches.take(1)

        name = "resnet50_delme"
        export_path = os.path.join(name, "export")
        backbone_export_path = os.path.join(export_path, "backbone")
        head_export_path = os.path.join(export_path, "head")

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        main(resnext_backbone, name, train_batches, test_batches, 1)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss1 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss1 = loss_fn(test_batch[1], classes)

        main(resnext_backbone, name, train_batches, test_batches, 5)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss2 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss2 = loss_fn(test_batch[1], classes)

        self.assertLess(loss2, loss1)
        shutil.rmtree(name)

    def test_resnet18(self):
        nf = 16
        bs = 4
        resnext_backbone = get_resnet18_backbone(nf)

        train_batches, _ = get_data(bs)
        train_batches = train_batches.take(10)
        test_batches = train_batches.take(1)

        name = "resnet18_delme"
        export_path = os.path.join(name, "export")
        backbone_export_path = os.path.join(export_path, "backbone")
        head_export_path = os.path.join(export_path, "head")

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        main(resnext_backbone, name, train_batches, test_batches, 1)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss1 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss1 = loss_fn(test_batch[1], classes)

        main(resnext_backbone, name, train_batches, test_batches, 5)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss2 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss2 = loss_fn(test_batch[1], classes)

        self.assertLess(loss2, loss1)
        shutil.rmtree(name)


class TestXResNet(TestCase):
    def test_xresnet50(self):
        nf = 16
        bs = 4
        resnext_backbone = get_xresnet50_backbone(nf)

        train_batches, _ = get_data(bs)
        train_batches = train_batches.take(10)
        test_batches = train_batches.take(1)

        name = "resnet50_delme"
        export_path = os.path.join(name, "export")
        backbone_export_path = os.path.join(export_path, "backbone")
        head_export_path = os.path.join(export_path, "head")

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        main(resnext_backbone, name, train_batches, test_batches, 1)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss1 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss1 = loss_fn(test_batch[1], classes)

        main(resnext_backbone, name, train_batches, test_batches, 5)
        backbone = tf.keras.models.load_model(backbone_export_path)
        head = tf.keras.models.load_model(head_export_path)
        loss2 = 0
        for test_batch in test_batches:
            embedding = backbone(test_batch[0])
            classes = head(embedding)
            loss2 = loss_fn(test_batch[1], classes)

        self.assertLess(loss2, loss1)
        shutil.rmtree(name)
