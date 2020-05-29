import tensorflow_datasets
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# Construct a tf.data.Dataset
raw_train = tensorflow_datasets.load('imagenette/160px', split='train')  # 12,894
raw_validation = tensorflow_datasets.load('imagenette/160px', split='validation')  # 500


# WARNING it seems that this validation set differs from original imagenette validation.

def format_example(sample):
    image = sample['image']
    label = sample['label']
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.random_crop(image, (IMG_SIZE, IMG_SIZE, 3))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_batches = validation.batch(BATCH_SIZE)
