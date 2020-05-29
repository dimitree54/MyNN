import tensorflow as tf


def resize_by_shorter_size(image: tf.Tensor, size: [int, tuple]):
    """
    :param image: HWC image tensor
    :param size: int to resize to exact size, tuple (int, int) to random sample size
    :return: augmented image
    """
    if isinstance(size, int):
        target_shorter_size = tf.cast(size, tf.float32)
    else:
        target_shorter_size = tf.random.uniform((), size[0], size[1], tf.float32)

    initial_width = tf.cast(tf.shape(image)[0], tf.float32)
    initial_height = tf.cast(tf.shape(image)[1], tf.float32)
    shorter_size = tf.minimum(initial_width, initial_height)
    ratio = shorter_size / target_shorter_size

    new_width = tf.cast(initial_width / ratio, tf.int32)
    new_height = tf.cast(initial_height / ratio, tf.int32)

    resized_image = tf.image.resize(image, (new_height, new_width))
    return resized_image
