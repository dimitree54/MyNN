from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow.keras import Model


class ModelBuilder(ABC):
    @abstractmethod
    def build(self, **kwargs) -> Model:
        pass


class ReLUBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.ReLU()
        ], **kwargs)


class MaxPoolBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D()
        ], **kwargs)


class AvgPoolBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.AvgPool2D()
        ], **kwargs)


class SumBlockBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Add(**kwargs)
        ], **kwargs)


class ConcatBlockBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Concatenate(**kwargs)
        ], **kwargs)


class ClassificationHeadBuilder(ModelBuilder):
    def build(self, num_classes, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(num_classes)
        ])


class ClassificationModel(Model):
    def __init__(self, backbone: Model, head: Model, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.head(x)
        return x
