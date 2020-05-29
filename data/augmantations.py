import tensorflow as tf


def resize_by_shorter_size(image: tf.Tensor, size: [int, tuple]):
    """
    :param image: HWC image tensor
    :param size: int to resize to exact size, tuple (int, int) to random sample size
    :return: augmented image
    """
    if isinstance(size, int):
        target_shorter_size = size
    else:
        target_shorter_size = tf.random.uniform((), size[0], size[1], tf.int32)
    shorter_size = tf.reduce_min(image.shape[:2])
    larger_size = tf.cast(tf.math.round(
        tf.reduce_max(image.shape[:2]) * target_shorter_size / shorter_size), tf.int32)
    resized_image = tf.image.resize(tf.expand_dims(image, 0), )
    pass
