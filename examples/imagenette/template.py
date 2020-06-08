import os
import tensorflow as tf

from misc.callbacks import ModelCheckpointBestAndLast, add_warm_up_to_lr
from models.base_classes import ClassificationHeadBuilder, ClassificationModel


def get_epoch_from_checkpoint_name(name):
    without_path = os.path.split(name)[1]
    return int(without_path)


def main(backbone, name, train_batches, validation_batches, epochs=200):
    logs_dir = name
    checkpoint_path = os.path.join(logs_dir, "{epoch}")
    tensorboard_path = os.path.join(logs_dir, "tensorboard")
    export_path = os.path.join(logs_dir, "export")
    backbone_export_path = os.path.join(export_path, "backbone")
    head_export_path = os.path.join(export_path, "head")

    head = ClassificationHeadBuilder().build(10)
    model = ClassificationModel(backbone, head)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(add_warm_up_to_lr(
        10, tf.keras.experimental.CosineDecay(0.1, epochs)))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    saver = ModelCheckpointBestAndLast(checkpoint_path)

    latest_checkpoint = tf.train.latest_checkpoint(logs_dir)
    if latest_checkpoint:
        prev_epoch = get_epoch_from_checkpoint_name(latest_checkpoint)
    else:
        prev_epoch = 0

    model.compile(tf.keras.optimizers.SGD(0.1, momentum=0.9),
                  tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                  tf.keras.metrics.SparseCategoricalAccuracy())

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.fit(train_batches, epochs=epochs, callbacks=[lr_schedule, tensorboard, saver],
              initial_epoch=prev_epoch, validation_data=validation_batches)

    backbone.save(backbone_export_path, include_optimizer=False)  # TODO test transfer learning restoring this backbone
    head.save(head_export_path, include_optimizer=False)
