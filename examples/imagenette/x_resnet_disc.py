import os

import tensorflow as tf
from tqdm import tqdm

from datasets.images.classification.imagenette import get_data_raw, val_preprocess, restore, resize_crop_augmentation, \
    parametrized_augmentation_transform, parametrized_extra_augmentation_transform, preprocess, SHUFFLE_BUFFER_SIZE
from misc.callbacks import add_warm_up_to_lr
from models.convolutional.architectures.decoders import get_x_resnet50_decoder
from models.convolutional.architectures.xresnet import get_x_resnet50_backbone
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


def scale_data(data, new_min, new_max):
    current_min = tf.reduce_min(data)
    current_max = tf.reduce_max(data)
    x_normed = (data - current_min) / (current_max - current_min)
    x_normed = x_normed * (new_max - new_min) + new_min
    return x_normed


def resize_crop_augmentation_wrapper(sample):
    image = sample['image']
    label = sample['label']
    image = resize_crop_augmentation(image)
    return {'image': image, 'label': label}


def train(epochs):
    ckpt.restore(manager.latest_checkpoint)
    for epoch in tqdm(range(int(step.numpy()), epochs)):
        lr_getter.set_epoch(epoch)
        for train_batch in train_batches:
            label = train_batch['label']
            label = tf.one_hot(label, 10, 1, 0, -1, tf.float32)
            # normal samples
            image = train_batch['image']
            image = parametrized_augmentation_transform(image, 1)
            image = preprocess(image)
            train_class_step((image, label))
            # hard samples
            image = train_batch['image']
            image = parametrized_augmentation_transform(image, aug_strength)
            image = parametrized_extra_augmentation_transform(image, aug_strength)
            image = preprocess(image)
            train_disc_step((image, label))

        with train_summary_writer.as_default():
            tf.summary.scalar("aug_strength", aug_strength, epoch)
            for train_batch in train_batches.take(1):
                image = train_batch['image']
                image = parametrized_augmentation_transform(image, aug_strength)
                image = parametrized_extra_augmentation_transform(image, aug_strength)
                image = preprocess(image)
                endpoints = xresnet_backbone(image, training=False)
                feedback = decoder(endpoints, training=False)
                tf.summary.image("aug_example", restore(image), epoch, 3)
                tf.summary.image("feedback", scale_data(feedback, 0, 1), epoch, 3)

        if class_accuracy_on_hard_samples.result() > 0.5:
            aug_strength.assign(tf.clip_by_value(tf.add(aug_strength, 0.1), 0, 2))
        else:
            aug_strength.assign(tf.clip_by_value(tf.subtract(aug_strength, 0.1), 0, 2))

        with train_summary_writer.as_default():
            for class_metric in class_metrics:
                tf.summary.scalar(class_metric.name, class_metric.result(), epoch)
                class_metric.reset_states()
            for disc_metric in disc_metrics:
                tf.summary.scalar(disc_metric.name, disc_metric.result(), epoch)
                disc_metric.reset_states()
            tf.summary.scalar("class_accuracy_on_hard_samples", class_accuracy_on_hard_samples.result(), epoch)
            class_accuracy_on_hard_samples.reset_states()
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
def train_class_step(train_batch):
    with tf.GradientTape() as tape:
        endpoints = xresnet_backbone(train_batch[0], training=True)
        class_logits = head(endpoints[-1], training=True)
        class_loss = class_loss_obj(train_batch[1], class_logits)
    class_gradients = tape.gradient(class_loss, xresnet_backbone.trainable_variables + head.trainable_variables)
    optimizer1.apply_gradients(zip(class_gradients, xresnet_backbone.trainable_variables + head.trainable_variables))

    for class_metric in class_metrics:
        class_metric.update_state(train_batch[1], class_logits)


@tf.function
def train_disc_step(train_batch):
    endpoints = xresnet_backbone(train_batch[0], training=True)
    class_logits = head(endpoints[-1], training=True)

    class_pred_label = tf.argmax(class_logits, -1)
    class_true_label = tf.argmax(train_batch[1], -1)
    class_correct = tf.cast(class_pred_label == class_true_label, tf.float32)

    with tf.GradientTape() as tape:
        feedback = decoder(endpoints, training=True)
        disc_embedding = disc_backbone(tf.concat([train_batch[0], feedback], -1), training=True)
        disc_logits = disc_head(disc_embedding, training=True)

        disc_prediction = tf.squeeze(tf.keras.activations.sigmoid(disc_logits))

        # for proper weighting we can not use tensorflow losses (because it reduces batch dimension, so
        # we calculate binary cross-entropy manually:
        disc_loss = disc_loss_obj(class_correct, disc_prediction)
    disc_gradient = tape.gradient(disc_loss, decoder.trainable_variables + disc_backbone.trainable_variables +
                                  disc_head.trainable_variables)
    optimizer2.apply_gradients(zip(disc_gradient, decoder.trainable_variables +
                                   disc_backbone.trainable_variables + disc_head.trainable_variables))
    for disc_metric in disc_metrics:
        disc_metric.update_state(class_correct, disc_prediction)
    class_accuracy_on_hard_samples.update_state(train_batch[1], class_logits)


if __name__ == "__main__":
    nf = 64
    bs = 16
    name = "x_resnet50_disc_aug"

    xresnet_backbone = get_x_resnet50_backbone(nf, return_endpoints_on_call=True)
    head = ClassificationHeadBuilder().build(10)
    decoder = get_x_resnet50_decoder(nf)
    disc_backbone = get_x_resnet50_backbone(nf)
    disc_head = ClassificationHeadBuilder().build(1)

    aug_strength = tf.Variable(0.0, trainable=False)
    train_batches, validation_batches = get_data_raw()
    train_batches = train_batches.map(resize_crop_augmentation_wrapper)
    validation_batches = validation_batches.map(val_preprocess)
    train_batches = train_batches.shuffle(SHUFFLE_BUFFER_SIZE).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = validation_batches.batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    lr_getter = LrGetter(add_warm_up_to_lr(10, tf.keras.experimental.CosineDecay(0.1, 200)))
    optimizer1 = tf.keras.optimizers.SGD(learning_rate=lr_getter.lr, momentum=0.9)
    optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr_getter.lr, momentum=0.9)
    class_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    disc_loss_obj = tf.keras.losses.BinaryCrossentropy()

    # call metrics and write to summary
    class_metrics = [tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)]
    class_accuracy_on_hard_samples = tf.keras.metrics.CategoricalAccuracy()
    disc_metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryCrossentropy(),
                    tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    step = tf.Variable(0)
    ckpt = tf.train.Checkpoint(step=step, optimizer1=optimizer1, optimizer2=optimizer2,
                               xresnet_backbone=xresnet_backbone, aug_strength=aug_strength,
                               head=head, decoder=decoder, disc_backbone=disc_backbone, disc_head=disc_head)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(name, 'ckpt'), max_to_keep=3)

    train(epochs=200)
    # TODO export (not only classifier backbone, but all models)
    # TODO we are trying to repeat normal classification for class_backbone + head, and applying modifications only to
    #  discriminator, but somehow validation accuracy not stable (compared to classical setting). where the difference
