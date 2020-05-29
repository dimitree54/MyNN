from models.architectures.resnet import *
from datasets.imagenet import train_batches, validation_batches


class ClassificationModel(Sequential):
    def __init__(self, cnn_backbone: Model, head: Model):
        super().__init__([
            cnn_backbone,
            head
        ])


if __name__ == "__main__":
    backbone = get_resnet18_backbone(64)
    classification_head = ClassificationHead(1000)
    class_model = ClassificationModel(
        cnn_backbone=backbone,
        head=classification_head)
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()
    tensorboard = tf.keras.callbacks.TensorBoard()
    class_model.compile(tf.keras.optimizers.SGD(0.1, momentum=0.9),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.metrics.SparseCategoricalAccuracy(), run_eagerly=True)
    class_model.fit(train_batches, epochs=10, callbacks=[lr_schedule, tensorboard],
                    validation_data=validation_batches)
    backbone.save("backbone")
    classification_head.save("backbone")
