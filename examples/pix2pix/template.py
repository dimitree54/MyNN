import os

import tensorflow as tf
from tqdm import tqdm

from datasets.imagenette import get_data, restore, get_data_raw, val_preprocess, resize_crop_augmentation, \
    parametrized_augmentation_transform, preprocess
from models.architectures.decoders import get_resnet18_decoder, get_resnet18_decoder_transposed
from models.architectures.resnet import get_resnet18_with_bottleneck_backbone
from models.base_classes import GeneratorModel


def train_preprocess(sample):
    image = sample['image']
    label = sample['label']
    label = tf.one_hot(label, 10, 1, 0, -1, tf.float32)
    image = resize_crop_augmentation(image)
    image = parametrized_augmentation_transform(image, 0.25)
    image = preprocess(image)
    return image, label


def train_resnet18():
    nf = 64
    bs = 16
    name = "resnet18_transposed"

    resnet_backbone = get_resnet18_with_bottleneck_backbone(nf, True)
    decoder = get_resnet18_decoder_transposed(nf)
    generator = GeneratorModel(resnet_backbone, decoder)
    discriminator = get_resnet18_with_bottleneck_backbone(nf)

    # default aug for classification too hard for color restoration
    train_batches, validation_batches = get_data_raw()
    train_batches = train_batches.map(train_preprocess)
    validation_batches = validation_batches.map(val_preprocess)
    train_batches = train_batches.shuffle(1000).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = validation_batches.batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    train(generator, discriminator, name, train_batches, validation_batches)


def grayscale(images_batch):
    return tf.image.rgb_to_grayscale(images_batch)


def train(generator, discriminator, name, train_batches, validation_batches, epochs=200):
    metrics = {
        "gen_loss_from_disc": tf.keras.metrics.Mean(),
        "gen_l1_loss": tf.keras.metrics.Mean(),
        "total_gen_loss": tf.keras.metrics.Mean(),
        "disc_real_loss": tf.keras.metrics.Mean(),
        "disc_generated_loss": tf.keras.metrics.Mean(),
        "total_disc_loss": tf.keras.metrics.Mean(),
        "disc_fake_accuracy": tf.keras.metrics.BinaryAccuracy(),
        "disc_real_accuracy": tf.keras.metrics.BinaryAccuracy()
    }
    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    step = tf.Variable(0)
    ckpt = tf.train.Checkpoint(step=step, generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(name, 'ckpt'), max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)

    for epoch in tqdm(range(int(step.numpy()), epochs)):
        for train_batch in train_batches:
            colored_images = train_batch[0]
            gray_images = grayscale(colored_images)
            train_step(generator, discriminator, gray_images, colored_images, metrics)
        with train_summary_writer.as_default():
            for metric_name in metrics:
                tf.summary.scalar(metric_name, metrics[metric_name].result(), epoch)
                metrics[metric_name].reset_states()
            for train_batch in train_batches.take(1):
                colored_images = train_batch[0]
                gray_images = grayscale(colored_images)
                generated_colored_images = generator(gray_images, training=False)
                tf.summary.image("gray_images", restore(gray_images), epoch, 3)
                tf.summary.image("colored_images", restore(colored_images), epoch, 3)
                tf.summary.image("generated_colored_images", restore(generated_colored_images), epoch, 3)
        for val_batch in validation_batches:
            colored_images = val_batch[0]
            gray_images = grayscale(colored_images)
            val_step(generator, discriminator, gray_images, colored_images, metrics)
        with val_summary_writer.as_default():
            for metric_name in metrics:
                tf.summary.scalar(metric_name, metrics[metric_name].result(), epoch)
                metrics[metric_name].reset_states()
            for val_batch in validation_batches.take(1):
                colored_images = val_batch[0]
                gray_images = grayscale(colored_images)
                generated_colored_images = generator(gray_images, training=False)
                tf.summary.image("gray_images", restore(gray_images), epoch, 3)
                tf.summary.image("colored_images", restore(colored_images), epoch, 3)
                tf.summary.image("generated_colored_images", restore(generated_colored_images), epoch, 3)
        step.assign_add(1)
        manager.save()


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(generator, discriminator, input_image, target, metrics):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        with tf.GradientTape() as disc_tape:  # discriminator do not have to know about generator gradients
            disc_real_output = discriminator(tf.concat([input_image, target], -1), training=True)
            disc_generated_output = discriminator(tf.concat([input_image, gen_output], -1), training=True)

            gen_loss_from_disc = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
            gen_l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
            total_gen_loss = gen_loss_from_disc + (100 * gen_l1_loss)

            disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
            disc_generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
            total_disc_loss = (disc_real_loss + disc_generated_loss)

    generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    metrics['gen_loss_from_disc'].update_state(gen_loss_from_disc)
    metrics['gen_l1_loss'].update_state(gen_l1_loss)
    metrics['total_gen_loss'].update_state(total_gen_loss)
    metrics['disc_real_loss'].update_state(disc_real_loss)
    metrics['disc_generated_loss'].update_state(disc_generated_loss)
    metrics['total_disc_loss'].update_state(total_disc_loss)
    metrics["disc_fake_accuracy"].update_state(tf.zeros_like(disc_generated_output), disc_generated_output)
    metrics["disc_real_accuracy"].update_state(tf.ones_like(disc_real_output), disc_real_output)


@tf.function
def val_step(generator, discriminator, input_image, target, metrics):
    gen_output = generator(input_image, training=False)
    disc_real_output = discriminator(tf.concat([input_image, target], -1), training=False)
    disc_generated_output = discriminator(tf.concat([input_image, gen_output], -1), training=False)

    gen_loss_from_disc = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    gen_l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gen_loss_from_disc + (100 * gen_l1_loss)

    disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    disc_generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = (disc_real_loss + disc_generated_loss) / 2

    metrics['gen_loss_from_disc'].update_state(gen_loss_from_disc)
    metrics['gen_l1_loss'].update_state(gen_l1_loss)
    metrics['total_gen_loss'].update_state(total_gen_loss)
    metrics['disc_real_loss'].update_state(disc_real_loss)
    metrics['disc_generated_loss'].update_state(disc_generated_loss)
    metrics['total_disc_loss'].update_state(total_disc_loss)
    metrics["disc_fake_accuracy"].update_state(tf.zeros_like(disc_generated_output), disc_generated_output)
    metrics["disc_real_accuracy"].update_state(tf.ones_like(disc_real_output), disc_real_output)


train_resnet18()
