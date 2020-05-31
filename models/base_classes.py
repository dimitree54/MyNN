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


class SumBlockBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Add(**kwargs)
        ], **kwargs)


class ClassificationHeadBuilder(ModelBuilder):
    def build(self, num_classes, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(num_classes)
        ])


class ClassificationModel(tf.keras.Sequential):
    def __init__(self, backbone: Model, head: Model):
        super().__init__([
            backbone,
            head
        ])
