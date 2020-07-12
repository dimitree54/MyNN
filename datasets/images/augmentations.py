import tensorflow as tf


def add_gaussian_noise(image, stddev):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    noisy_image = tf.add(image, noise)
    return noisy_image
