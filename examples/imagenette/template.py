import os
import tensorflow as tf

from models.base_classes import ClassificationHeadBuilder, ClassificationModel
from datasets.imagenette import get_data


def get_epoch_from_checkpoint_name(name):
    without_path = os.path.split(name)[1]
    return int(without_path)


def main(backbone, name, bs):
    logs_dir = os.path.join(name)
    checkpoint_path = os.path.join(logs_dir, "{epoch}")
    tensorboard_path = os.path.join(logs_dir, "tensorboard")
    export_path = os.path.join(logs_dir, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    head_export_path = os.path.join(export_path, "head")

    train_batches, validation_batches = get_data(bs)

    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(backbone, head)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    saver = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True)

    latest_checkpoint = tf.train.latest_checkpoint(logs_dir)
    if latest_checkpoint:
        prev_epoch = get_epoch_from_checkpoint_name(latest_checkpoint)
    else:
        prev_epoch = 0

    model.compile(tf.keras.optimizers.SGD(0.1, momentum=0.9),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  tf.keras.metrics.SparseCategoricalAccuracy())

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.fit(train_batches, epochs=1000, callbacks=[lr_schedule, tensorboard, saver],
              initial_epoch=prev_epoch, validation_data=validation_batches)

    backbone.save(backbone_export_path, include_optimizer=False)
    head.save(head_export_path, include_optimizer=False)
