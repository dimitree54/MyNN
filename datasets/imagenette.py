import tensorflow_datasets
import tensorflow as tf

IMG_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000

# Construct a tf.data.Dataset
raw_train, info = tensorflow_datasets.load('imagenette/160px', split='train', with_info=True)  # 12,894
raw_validation = tensorflow_datasets.load('imagenette/160px', split='validation')  # 500


# WARNING it seems that this validation set differs from original imagenette validation.

def format_example(sample):
    image = sample['image']
    label = sample['label']
    image = aug(image)
    image = preprocess(image)
    return image, label


def aug(image):
    image = tf.image.random_crop(image, (IMG_SIZE, IMG_SIZE, 3))
    return image


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def restore(image):
    image = (image + 1) * 127.5
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image


def get_data(batch_size):
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = validation.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches


def draw_examples(data_batches, n=3):
    import matplotlib.pyplot as plt
    for sample in data_batches.take(n):
        image = restore(sample[0])[0]
        plt.imshow(image)
        plt.show()
        print(info.features["label"].int2str(sample[1][0]))
