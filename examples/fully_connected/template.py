import os
import tensorflow as tf
from tqdm import tqdm

from models.fully_connected.local import LocalFCLayerWithExternalOutput

default_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def fake_loss_object(true, pred):  # noqa
    return 0


metrics = {
    "loss": tf.keras.metrics.Mean(),
    "model_loss": tf.keras.metrics.Mean(),
    "total_loss": tf.keras.metrics.Mean(),
    "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
    "mean_output_activation": tf.keras.metrics.Mean(),
}


@tf.function
def train_step(model: tf.keras.Model, optimizer, loss_object, x, y, num_classes):
    if isinstance(model.layers[-1], LocalFCLayerWithExternalOutput):
        one_hot_y = tf.one_hot(y, num_classes, 1, 0, dtype=tf.float32)
        model.layers[-1].set_output(one_hot_y)
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_object(y, [pred])
        if len(model.losses) == 0:
            model_loss = 0
        else:
            model_loss = tf.add_n(model.losses)
        total_loss = loss_value + model_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metrics["loss"].update_state(loss_value)
    metrics["model_loss"].update_state(model_loss)
    metrics["total_loss"].update_state(total_loss)
    metrics["accuracy"].update_state(y, pred)
    metrics["mean_output_activation"].update_state(tf.reduce_mean(tf.abs(pred)))


@tf.function
def val_step(model, loss_object, x, y):
    pred = model(x, training=False)
    loss_value = loss_object(y, [pred])
    if len(model.losses) == 0:
        model_loss = 0
    else:
        model_loss = tf.add_n(model.losses)
    total_loss = loss_value + model_loss

    metrics["loss"].update_state(loss_value)
    metrics["model_loss"].update_state(model_loss)
    metrics["total_loss"].update_state(total_loss)
    metrics["accuracy"].update_state(y, pred)
    metrics["mean_output_activation"].update_state(tf.reduce_mean(tf.abs(pred)))


def train(model: tf.keras.Model, name, train_batches, validation_batches, epochs, base_lr,
          loss_object=default_loss_object):
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    num_classes = model.output_shape[-1]

    for epoch in tqdm(range(epochs)):
        for batch in train_batches:
            train_step(model, optimizer, loss_object, batch['features'], batch['label'], num_classes)
        with train_summary_writer.as_default():
            for metric_name in metrics:
                tf.summary.scalar(metric_name, metrics[metric_name].result(), epoch)
                metrics[metric_name].reset_states()

        for batch in validation_batches:
            val_step(model, loss_object, batch['features'], batch['label'])
        with val_summary_writer.as_default():
            for metric_name in metrics:
                tf.summary.scalar(metric_name, metrics[metric_name].result(), epoch)
                metrics[metric_name].reset_states()
