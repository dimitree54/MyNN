import os

import tensorflow as tf
from tqdm import tqdm

from datasets.imagenette import get_data
from misc.callbacks import add_warm_up_to_lr
from models.architectures.decoders import get_x_resnet50_decoder
from models.architectures.xresnet import get_x_resnet50_backbone
from models.base_classes import ClassificationHeadBuilder


class LrGetter:
    def __init__(self, fn):
        self.fn = fn
        self.epoch = 0
        self.lr = tf.Variable(0.0, trainable=False)

    def get(self):
        return self.fn(self.epoch)

    def set_epoch(self, epoch):
        self.lr.assign(self.fn(epoch))


def train(epochs):
    ckpt.restore(manager.latest_checkpoint)
    for epoch in tqdm(range(int(step.numpy()), epochs)):
        lr_getter.set_epoch(epoch)
        for train_batch in train_batches:
            train_step(train_batch)

        class_acc.assign(class_metrics[0].result())

        with train_summary_writer.as_default():
            for class_metric in class_metrics:
                tf.summary.scalar(class_metric.name, class_metric.result(), epoch)
                class_metric.reset_states()
            for disc_metric in disc_metrics:
                tf.summary.scalar(disc_metric.name, disc_metric.result(), epoch)
                disc_metric.reset_states()
            tf.summary.scalar("lr", lr_getter.lr, epoch)

        for val_batch in validation_batches:
            val_step(val_batch)

        with val_summary_writer.as_default():
            for class_metric in class_metrics:
                tf.summary.scalar(class_metric.name, class_metric.result(), epoch)
                class_metric.reset_states()
            for disc_metric in disc_metrics:
                tf.summary.scalar(disc_metric.name, disc_metric.result(), epoch)
                disc_metric.reset_states()
        step.assign_add(1)
        manager.save()


@tf.function
def val_step(val_batch):
    endpoints = xresnet_backbone(val_batch[0])
    class_logits = head(endpoints[-1])
    feedback = decoder(endpoints)
    disc_embedding = disc_backbone(tf.concat([val_batch[0], feedback], -1))
    disc_logits = disc_head(disc_embedding)
    class_pred_label = tf.argmax(class_logits, -1)
    class_true_label = tf.argmax(val_batch[1], -1)
    class_correct = class_pred_label == class_true_label
    disc_prediction = tf.keras.activations.sigmoid(disc_logits)
    for class_metric in class_metrics:
        class_metric.update_state(val_batch[1], class_logits)
    for disc_metric in disc_metrics:
        disc_metric.update_state(class_correct, disc_prediction)


@tf.function
def train_step(train_batch):
    with tf.GradientTape() as tape:
        endpoints = xresnet_backbone(train_batch[0], training=True)
        class_logits = head(endpoints[-1], training=True)
        class_loss = class_loss_obj(train_batch[1], class_logits)
    class_gradients = tape.gradient(class_loss, xresnet_backbone.trainable_variables + head.trainable_variables)
    optimizer1.apply_gradients(zip(class_gradients, xresnet_backbone.trainable_variables + head.trainable_variables))

    class_pred_label = tf.argmax(class_logits, -1)
    class_true_label = tf.argmax(train_batch[1], -1)
    class_correct = tf.cast(class_pred_label == class_true_label, tf.float32)

    weight_for_0 = (1 - class_acc) / 2.0
    weight_for_1 = class_acc / 2.0
    loss_weights = class_correct * weight_for_1 + (1 - class_correct) * weight_for_0

    with tf.GradientTape() as tape:
        feedback = decoder(endpoints, training=True)
        disc_embedding = disc_backbone(tf.concat([train_batch[0], feedback], -1), training=True)
        disc_logits = disc_head(disc_embedding, training=True)

        disc_prediction = tf.squeeze(tf.keras.activations.sigmoid(disc_logits))

        # for proper weighting we can not use tensorflow losses (because it reduces batch dimension, so
        # we calculate binary cross-entropy manually:
        disc_loss = (class_correct * tf.math.log(disc_prediction + 0.000001)) + \
                    ((1 - class_correct) * tf.math.log(1 - disc_prediction + 0.000001))
        disc_loss = -tf.reduce_mean(disc_loss * loss_weights)
    disc_gradient = tape.gradient(disc_loss, decoder.trainable_variables + disc_backbone.trainable_variables +
                                  disc_head.trainable_variables)
    optimizer2.apply_gradients(zip(disc_gradient, decoder.trainable_variables +
                                   disc_backbone.trainable_variables + disc_head.trainable_variables))

    for class_metric in class_metrics:
        class_metric.update_state(train_batch[1], class_logits)
    for disc_metric in disc_metrics:
        disc_metric.update_state(class_correct, disc_prediction)


if __name__ == "__main__":
    nf = 64
    bs = 16
    name = "x_resnet50_disc"

    xresnet_backbone = get_x_resnet50_backbone(nf, return_endpoints_on_call=True)
    head = ClassificationHeadBuilder().build(10)
    decoder = get_x_resnet50_decoder(nf)
    disc_backbone = get_x_resnet50_backbone(nf)
    disc_head = ClassificationHeadBuilder().build(1)

    train_batches, validation_batches = get_data(bs)

    lr_getter = LrGetter(add_warm_up_to_lr(10, tf.keras.experimental.CosineDecay(0.1, 200)))
    optimizer1 = tf.keras.optimizers.SGD(learning_rate=lr_getter.lr, momentum=0.9)
    optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr_getter.lr, momentum=0.9)
    class_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

    # call metrics and write to summary
    class_metrics = [tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)]
    disc_metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryCrossentropy(),
                    tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    class_acc = tf.Variable(0.5)
    step = tf.Variable(0)
    ckpt = tf.train.Checkpoint(step=step, optimizer1=optimizer1, optimizer2=optimizer2,
                               xresnet_backbone=xresnet_backbone, class_acc=class_acc,
                               head=head, decoder=decoder, disc_backbone=disc_backbone, disc_head=disc_head)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(name, 'ckpt'), max_to_keep=3)

    train(epochs=200)
    # TODO export (not only classifier backbone, but all models)
    # TODO add feedback images summary saving.
