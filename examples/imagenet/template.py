import os

import tensorflow as tf

from misc.callbacks import ModelCheckpointBestAndLast, add_warm_up_to_lr


def get_epoch_from_checkpoint_name(name):
    without_path = os.path.split(name)[1]
    return int(without_path)


def train(model: tf.keras.Model, name, train_batches, validation_batches, epochs=120, base_lr=0.1):
    logs_dir = name
    checkpoint_path = os.path.join(logs_dir, "{epoch}")
    tensorboard_path = os.path.join(logs_dir, "tensorboard")

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(add_warm_up_to_lr(
        10, tf.keras.experimental.CosineDecay(base_lr, epochs)))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    saver = ModelCheckpointBestAndLast(checkpoint_path)

    latest_checkpoint = tf.train.latest_checkpoint(logs_dir)
    if latest_checkpoint:
        prev_epoch = get_epoch_from_checkpoint_name(latest_checkpoint)
    else:
        prev_epoch = 0

    model.compile(tf.keras.optimizers.SGD(momentum=0.9),
                  tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                  tf.keras.metrics.CategoricalAccuracy())

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.fit(train_batches, epochs=epochs, callbacks=[lr_schedule, tensorboard, saver],
              initial_epoch=prev_epoch, validation_data=validation_batches)
