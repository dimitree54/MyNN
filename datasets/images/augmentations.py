import tensorflow as tf


def add_gaussian_noise(image, stddev):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    noisy_image = tf.add(image, noise)
    return noisy_image


def resize_by_shorter_size(image: tf.Tensor, target_shorter_size: [int, tuple]):
    """
    :param image: HWC image tensor
    :param target_shorter_size: int to resize to exact size, tuple (int, int) to sample random size
    :return: augmented image
    """
    if isinstance(target_shorter_size, int):
        target_shorter_size = tf.cast(target_shorter_size, tf.float32)
    else:
        target_shorter_size = tf.random.uniform((), target_shorter_size[0], target_shorter_size[1], tf.float32)

    initial_height = tf.cast(tf.shape(image)[0], tf.float32)
    initial_width = tf.cast(tf.shape(image)[1], tf.float32)
    shorter_size = tf.minimum(initial_width, initial_height)
    ratio = shorter_size / target_shorter_size

    new_width = tf.cast(initial_width / ratio, tf.int32)
    new_height = tf.cast(initial_height / ratio, tf.int32)

    resized_image = tf.image.resize(image, (new_height, new_width))
    return resized_image


def random_crop_and_resize(image: tf.Tensor, target_size: tuple,
                           area_range: tuple, aspect_ratio_range: tuple):
    """
    :param image: HWC image tensor
    :param target_size: tuple (int, int), crop will be resized to target_size
    :param area_range: tuple (float, float) crop will have area rate (relative to input image area) from this range
    :param aspect_ratio_range: aspect ratio (width/height) of crop will be sampled from this range
    :return: augmented image
    """
    initial_height = tf.cast(tf.shape(image)[0], tf.float32)
    initial_width = tf.cast(tf.shape(image)[1], tf.float32)
    initial_area = initial_height * initial_width

    crop_area = tf.random.uniform((), area_range[0], area_range[1], tf.float32) * initial_area
    base_crop_size = tf.sqrt(crop_area)
    crop_aspect_ratio = tf.random.uniform((), aspect_ratio_range[0], aspect_ratio_range[1], tf.float32)
    crop_height = tf.minimum(base_crop_size / crop_aspect_ratio, initial_height)
    crop_width = tf.minimum(base_crop_size * crop_aspect_ratio, initial_width)

    cropped_image = tf.image.random_crop(image, (crop_height, crop_width, 3))
    cropped_and_resized_image = tf.image.resize(cropped_image, target_size)
    return cropped_and_resized_image
