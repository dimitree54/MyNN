import os
import tensorflow as tf

from models.architectures.resnet import get_resnet50_backbone
from models.base_classes import ClassificationHeadBuilder, ClassificationModel
from datasets.imagenette import get_data

NF = 64
BS = 32


def get_epoch_lr_from_checkpoint_name(name):
    without_path = os.path.split(name)[1]
    epoch_str, lr_str = without_path.split('-')
    return int(epoch_str), float(lr_str)


if __name__ == "__main__":
    logs_dir = os.path.join("resnet50")
    checkpoint_path = os.path.join(logs_dir, "{epoch}-{lr:.5f}")
    tensorboard_path = os.path.join(logs_dir, "tensorboard")
    export_path = os.path.join(logs_dir, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    head_export_path = os.path.join(export_path, "head")

    train_batches, validation_batches = get_data(BS)
    train_batches = train_batches
    validation_batches = validation_batches

    backbone = get_resnet50_backbone(NF)
    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(backbone, head)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    saver = tf.keras.callbacks.ModelCheckpoint(checkpoint_path)

    latest_checkpoint = tf.train.latest_checkpoint(logs_dir)
    if latest_checkpoint:
        prev_epoch, init_lr = get_epoch_lr_from_checkpoint_name(latest_checkpoint)
    else:
        prev_epoch, init_lr = 0, 0.1

    model.compile(tf.keras.optimizers.SGD(init_lr, momentum=0.9),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  tf.keras.metrics.SparseCategoricalAccuracy())

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    history = model.fit(train_batches, epochs=100, callbacks=[lr_schedule, tensorboard, saver],
                        initial_epoch=prev_epoch, validation_data=validation_batches)

    backbone.save(backbone_export_path)
    head.save(head_export_path)
