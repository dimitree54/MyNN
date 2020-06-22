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

    def get(self):
        return self.fn(self.epoch)


if __name__ == "__main__":
    nf = 4
    bs = 3
    name = "x_resnet50_disc"

    xresnet_backbone = get_x_resnet50_backbone(nf, return_endpoints_on_call=True)
    head = ClassificationHeadBuilder().build(10)
    decoder = get_x_resnet50_decoder(nf)
    disc_backbone = get_x_resnet50_backbone(nf)
    disc_head = ClassificationHeadBuilder().build(1)

    train_batches, validation_batches = get_data(bs)

    epoch = 0
    lr_getter = LrGetter(add_warm_up_to_lr(10, tf.keras.experimental.CosineDecay(0.1, 200)))
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_getter.get, momentum=0.9)
    class_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    disc_loss_obj = tf.keras.losses.BinaryCrossentropy()

    # call metrics and write to summary
    class_metrics_train = [tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)]
    disc_metrics_train = [tf.keras.metrics.BinaryAccuracy(), tf.keras.losses.BinaryCrossentropy(),
                          tf.keras.metrics.TruePositives, tf.keras.metrics.TrueNegatives,
                          tf.keras.metrics.FalsePositives, tf.keras.metrics.FalseNegatives]
    class_metrics_val = [tf.keras.metrics.CategoricalAccuracy(),
                         tf.keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)]
    disc_metrics_val = [tf.keras.metrics.BinaryAccuracy(), tf.keras.losses.BinaryCrossentropy(),
                        tf.keras.metrics.TruePositives, tf.keras.metrics.TrueNegatives,
                        tf.keras.metrics.FalsePositives, tf.keras.metrics.FalseNegatives]
    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    for epoch in tqdm(range(200)):
        lr_getter.epoch = epoch
        for train_batch in train_batches:  # TODO move training to tf.function
            with tf.GradientTape(persistent=True) as tape:
                endpoints = xresnet_backbone(train_batch[0])
                class_logits = head(endpoints[-1])
                feedback = decoder(endpoints)
                disc_embedding = disc_backbone(tf.concat([train_batch[0], feedback], -1))
                disc_logits = disc_head(disc_embedding)

                class_loss = class_loss_obj(train_batch[1], class_logits)
                class_pred_label = tf.argmax(class_logits, -1)
                class_true_label = tf.argmax(train_batch[1], -1)
                class_correct = class_pred_label == class_true_label

                disc_prediction = tf.keras.activations.sigmoid(disc_logits)
                disc_loss = disc_loss_obj(class_correct, disc_prediction)  # TODO weight this loss

            class_gradients = tape.gradient(class_loss, xresnet_backbone.trainable_variables + head.trainable_variables)
            disc_gradient = tape.gradient(disc_loss, decoder.trainable_variables + disc_backbone.trainable_variables +
                                          disc_head.trainable_variables)
            optimizer.apply_gradients(zip(class_gradients, xresnet_backbone.trainable_variables +
                                          head.trainable_variables))
            optimizer.apply_gradients(zip(disc_gradient, decoder.trainable_variables +
                                          disc_backbone.trainable_variables + disc_head.trainable_variables))

            for class_metric in class_metrics_train:
                class_metric.update_state(train_batch[1], class_logits)
            for disc_metric in disc_metrics_train:
                disc_metric.update_state(class_correct, disc_prediction)

        for val_batch in validation_batches:
            endpoints = xresnet_backbone(val_batch[0])
            class_logits = head(endpoints[-1])
            feedback = decoder(endpoints)
            disc_embedding = disc_backbone(tf.concat([val_batch[0], feedback], -1))
            disc_logits = disc_head(disc_embedding)

            class_loss = class_loss_obj(val_batch[1], class_logits)
            class_pred_label = tf.argmax(class_logits, -1)
            class_true_label = tf.argmax(val_batch[1], -1)
            class_correct = class_pred_label == class_true_label

            disc_prediction = tf.keras.activations.sigmoid(disc_logits)
            disc_loss = disc_loss_obj(class_correct, disc_prediction)

            for class_metric in class_metrics_val:
                class_metric.update_state(val_batch[1], class_logits)
            for disc_metric in disc_metrics_val:
                disc_metric.update_state(class_correct, disc_prediction)

        with train_summary_writer.as_default():
            for class_metric in class_metrics_train:
                tf.summary.scalar(class_metric.name, class_metric.result(), epoch)
                class_metric.reset_states()
            for disc_metric in disc_metrics_train:
                tf.summary.scalar(disc_metric.name, disc_metric.result(), epoch)
                disc_metric.reset_states()
            tf.summary.scalar("lr", lr_getter.get(), epoch)

        with val_summary_writer.as_default():
            for class_metric in class_metrics_val:
                tf.summary.scalar(class_metric.name, class_metric.result(), epoch)
                class_metric.reset_states()
            for disc_metric in disc_metrics_val:
                tf.summary.scalar(disc_metric.name, disc_metric.result(), epoch)
                disc_metric.reset_states()
    # TODO export (not only classifier backbone, but all models)
