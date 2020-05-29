from models.architectures.resnet import *
from datasets.imagenet import train_batches, validation_batches


class ClassificationModel(Sequential):
    def __init__(self, cnn_backbone: Model, head: Model):
        super().__init__([
            cnn_backbone,
            head
        ])


if __name__ == "__main__":
    intermediate_save_dir = "saved"
    backbone_save_dir = "backbone"
    classification_head_save_dir = "head"

    backbone = get_resnet50_backbone(64)
    classification_head = ClassificationHead(1000)
    class_model = ClassificationModel(
        cnn_backbone=backbone,
        head=classification_head)

    latest_checkpoint = tf.train.latest_checkpoint(intermediate_save_dir)
    if latest_checkpoint:
        classification_head.load_weights(latest_checkpoint)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()
    tensorboard = tf.keras.callbacks.TensorBoard()
    saver = tf.keras.callbacks.ModelCheckpoint(intermediate_save_dir)

    class_model.compile(tf.keras.optimizers.SGD(0.1, momentum=0.9),
                        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        tf.keras.metrics.SparseCategoricalAccuracy())
    class_model.fit(train_batches, epochs=60, callbacks=[lr_schedule, tensorboard, saver],
                    validation_data=validation_batches)

    backbone.save(backbone_save_dir)
    classification_head.save(classification_head_save_dir)
