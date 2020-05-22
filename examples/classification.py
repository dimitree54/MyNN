from tensorflow.keras import Model, Sequential
import tensorflow as tf

from models.architectures.resnet import get_resnet50_backbone, ClassificationHead
from datasets.imagenette import train_batches, validation_batches


class ClassificationModel(Sequential):
    def __init__(self, backbone: Model, head: Model):
        super().__init__([
            backbone,
            head
        ])


if __name__ == "__main__":
    class_model = ClassificationModel(
        backbone=get_resnet50_backbone(64),
        head=ClassificationHead(10))
    class_model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.metrics.SparseCategoricalAccuracy(), run_eagerly=True)
    class_model.fit(train_batches, epochs=10)
    class_model.evaluate(validation_batches)


