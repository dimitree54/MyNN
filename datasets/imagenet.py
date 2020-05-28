import tensorflow_datasets
import tensorflow as tf

IMG_SIZE = 128  # All images will be resized to 160x160
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
NGF = 64

# Construct a tf.data.Dataset
raw_train = tensorflow_datasets.load('imagenet2012', split='train')
raw_validation = tensorflow_datasets.load('imagenet2012', split='validation')


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
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
