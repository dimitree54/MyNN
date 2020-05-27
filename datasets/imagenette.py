import tensorflow_datasets as tfds
import tensorflow as tf

IMG_SIZE = 128  # All images will be resized to 160x160
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
NGF = 64

# Construct a tf.data.Dataset
raw_train = tfds.load('imagenette/160px', split='train')  # 12,894
raw_validation = tfds.load('imagenette/160px', split='validation')  # 500


# WARNING it seems that this validation set differs from original imagenette validation.

def format_example(sample):
    image = sample['image']
    label = sample['label']
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.random_crop(image, (IMG_SIZE, IMG_SIZE, 3))
    image = tf.image.resize(image, (224, 224))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)