import tensorflow_datasets
import tensorflow as tf
import numpy as np

from data.augmantations import resize_by_shorter_size

IMG_SIZE = 224
BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 1000
IMAGENET_MEAN_RGB = [123.68, 116.779, 103.939]

# Construct a tf.data.Dataset
raw_train, info = tensorflow_datasets.load('imagenet2012', split='train', with_info=True)
raw_validation = tensorflow_datasets.load('imagenet2012', split='validation')


def format_example(sample):
    image = sample['image']
    label = sample['label']
    image = aug(image)
    image = preprocess(image)
    return image, label


def aug(image):
    image = resize_by_shorter_size(image, (256, 480))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (IMG_SIZE, IMG_SIZE, 3))
    # original paper uses color augmentation based on PCA, but I replaced it with more simple:
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = image - np.reshape(IMAGENET_MEAN_RGB, (1, 1, 3))
    return image


def restore(image):
    image = image + np.reshape(IMAGENET_MEAN_RGB, (1, 1, 3))
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


train = raw_train.cache().map(format_example)
validation = raw_validation.cache().map(format_example)
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_batches = validation.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


def draw_examples(n=3):
    import matplotlib.pyplot as plt
    for sample in train_batches.take(n):
        image = restore(sample[0])[0]
        plt.imshow(image)
        plt.show()
        print(info.features["label"].int2str(sample[1][0]))
