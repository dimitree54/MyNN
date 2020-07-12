import os

import tensorflow as tf

from datasets.vectors.classification.iris import get_data

name = "baseline"
NUM_EPOCHS = 201
BATCH_SIZE = 32


def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_object(y, [pred])
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_metric.update_state(loss_value)
    accuracy_metric.update_state(y, pred)


def val_step(x, y):
    pred = model(x, training=False)
    loss_value = loss_object(y, [pred])
    loss_metric.update_state(loss_value)
    accuracy_metric.update_state(y, pred)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3)
])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_metric = tf.keras.metrics.Mean()
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

train_data, val_data = get_data(BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    for batch in train_data:
        train_step(batch['features'], batch['label'])
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss_metric.result(), epoch)
        tf.summary.scalar("accuracy", accuracy_metric.result(), epoch)
    loss_metric.reset_states()
    accuracy_metric.reset_states()

    for batch in val_data:
        train_step(batch['features'], batch['label'])
    with val_summary_writer.as_default():
        tf.summary.scalar("loss", loss_metric.result(), epoch)
        tf.summary.scalar("accuracy", accuracy_metric.result(), epoch)
    loss_metric.reset_states()
    accuracy_metric.reset_states()
