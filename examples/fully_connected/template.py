import os
import tensorflow as tf
from tqdm import tqdm

default_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = {
    "loss": tf.keras.metrics.Mean(),
    "model_loss": tf.keras.metrics.Mean(),
    "total_loss": tf.keras.metrics.Mean(),
    "accuracy": tf.keras.metrics.SparseCategoricalAccuracy()
}


@tf.function
def train_step(model: tf.keras.Model, optimizer, loss_object, x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_object(y, [pred])
        model_loss = tf.add_n(model.losses)
        total_loss = loss_value + model_loss
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metrics["loss"].update_state(loss_value)
    metrics["model_loss"].update_state(model_loss)
    metrics["total_loss"].update_state(total_loss)
    metrics["accuracy"].update_state(y, pred)


@tf.function
def val_step(model, loss_object, x, y):
    pred = model(x, training=False)
    loss_value = loss_object(y, [pred])
    model_loss = tf.add_n(model.losses)
    total_loss = loss_value + model_loss

    metrics["loss"].update_state(loss_value)
    metrics["model_loss"].update_state(model_loss)
    metrics["total_loss"].update_state(total_loss)
    metrics["accuracy"].update_state(y, pred)


def train(model: tf.keras.Model, name, train_batches, validation_batches, epochs=120, base_lr=0.01,
          loss_object=default_loss_object):
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(name, "train"))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(name, "val"))

    for epoch in tqdm(range(epochs)):
        for batch in train_batches:
            train_step(model, optimizer, loss_object, batch['features'], batch['label'])
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
