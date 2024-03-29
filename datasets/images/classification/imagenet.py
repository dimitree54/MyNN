import tensorflow_datasets
import tensorflow as tf
import numpy as np

from datasets.images.augmentations import add_gaussian_noise, resize_by_shorter_size, random_crop_and_resize

IMG_SIZE = 224
SHUFFLE_BUFFER_SIZE = 1000
IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]
IMAGENET_DEV_RGB = [58.293, 57.12, 57.375]


def train_preprocess(sample):
    image = sample['image']
    label = sample['label']
    label = tf.one_hot(label, 1000, 1, 0, -1, tf.float32)
    image = resize_crop_augmentation(image)
    image = augmentation_transform(image)
    image = preprocess(image)
    return image, label


def val_preprocess(sample):
    image = sample['image']
    label = sample['label']
    label = tf.one_hot(label, 1000, 1, 0, -1, tf.float32)
    image = validation_transform(image)
    image = preprocess(image)
    return image, label


def resize_crop_augmentation(image):
    image = resize_by_shorter_size(image, (256, 480))
    image = random_crop_and_resize(image, (IMG_SIZE, IMG_SIZE), (0.08, 1), (3/4, 4/3))
    return image


def augmentation_transform(image):
    return parametrized_augmentation_transform(image, 1)


def parametrized_augmentation_transform(image, parameter):
    # Train data input pipeline mainly from paper xResNet, but without PCA color augmentation
    image = tf.image.random_flip_left_right(image)

    amplitude = parameter * 0.4 + 0.000001  # can not be zero
    image = tf.image.random_hue(image, min(amplitude, 0.5))
    image = tf.image.random_saturation(image, max(0, 1 - amplitude), min(2, 1 + amplitude))
    image = tf.image.random_brightness(image, amplitude)
    return image


def parametrized_extra_augmentation_transform(image, parameter):
    amplitude = parameter * 0.4 + 0.000001  # can not be zero
    image = tf.image.random_contrast(image, 1 - amplitude, 1 + amplitude)
    image = add_gaussian_noise(image, parameter * 0.1 * 255)
    return image


def validation_transform(image):
    image = resize_by_shorter_size(image, 256)
    image = tf.image.crop_to_bounding_box(
        image, (tf.shape(image)[0] - IMG_SIZE) // 2, (tf.shape(image)[1] - IMG_SIZE) // 2, IMG_SIZE, IMG_SIZE)
    return image


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image - np.reshape(IMAGENET_MEAN_RGB, (1, 1, 3))) / IMAGENET_DEV_RGB
    return image


def restore(image):
    image = image * IMAGENET_DEV_RGB + np.reshape(IMAGENET_MEAN_RGB, (1, 1, 3))
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def get_data(batch_size, draw_examples=False):
    # Construct a tf.data.Dataset
    raw_train, info = tensorflow_datasets.load('imagenet2012', split='train', with_info=True)
    raw_validation = tensorflow_datasets.load('imagenet2012', split='validation')

    train = raw_train.map(train_preprocess)
    validation = raw_validation.map(val_preprocess)
    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = validation.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if draw_examples:
        draw(train_batches, info)
        draw(validation_batches, info)
    return train_batches, validation_batches


def get_data_raw():
    # Construct a tf.data.Dataset
    raw_train, info = tensorflow_datasets.load('imagenet2012', split='train', with_info=True)
    raw_validation = tensorflow_datasets.load('imagenet2012', split='validation')

    return raw_train, raw_validation


def draw(data_batches, info, n=3):
    import matplotlib.pyplot as plt
    for sample in data_batches.take(n):
        image = restore(sample[0])[0]
        plt.imshow(image)
        plt.show()
        print(info.features["label"].int2str(tf.argmax(sample[1][0])))
