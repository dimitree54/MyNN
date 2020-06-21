from abc import abstractmethod, ABC

import tensorflow as tf
from tensorflow.keras import Model


class ModelBuilder(ABC):
    @abstractmethod
    def build(self, **kwargs) -> Model:
        pass


class ReLUBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.ReLU()
        ], **kwargs)


class SigmoidBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Activation("sigmoid")
        ], **kwargs)


class MaxPoolBuilder(ModelBuilder):
    def build(self, kernel_size=2, stride=2, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride, padding="same")
        ], **kwargs)


class AvgPoolBuilder(ModelBuilder):
    def build(self, kernel_size=2, stride=2,  **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.AvgPool2D(pool_size=kernel_size, strides=stride, padding="same")
        ], **kwargs)


class GlobalAvgPoolBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D()
        ], **kwargs)


class FCBlockBuilder(ModelBuilder):
    def build(self, units, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Dense(units=units)
        ], **kwargs)


class SumBlockBuilder(ModelBuilder):
    class AddModel(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.add_layer = tf.keras.layers.Add(**kwargs)

        def call(self, inputs, training=None, mask=None):
            return self.add_layer(inputs)

    def build(self, **kwargs) -> Model:
        return SumBlockBuilder.AddModel(**kwargs)


class IdentityBlockBuilder(ModelBuilder):
    def build(self, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.Activation('linear')
        ])


class ConcatBlockBuilder(ModelBuilder):
    class ConcatModel(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.concat_layer = tf.keras.layers.Add(**kwargs)

        def call(self, inputs, training=None, mask=None):
            return self.concat_layer(inputs)

    def build(self, **kwargs) -> Model:
        return ConcatBlockBuilder.ConcatModel(**kwargs)


class ClassificationHeadBuilder(ModelBuilder):
    def build(self, num_classes, **kwargs) -> Model:
        return tf.keras.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(num_classes)
        ])


class ClassificationModel(Model):
    def __init__(self, backbone: Model, head: Model, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs, training=training, mask=mask)
        x = self.head(x, training=training, mask=mask)
        return x


class GenerationModel(Model):
    def __init__(self, encoder: Model, decoder: Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs, training=training, mask=mask)
        x = self.decoder(x, training=training, mask=mask)
        return x


class GenerationWithClassificationModel(Model):
    def __init__(self, encoder: Model, decoder: Model, classification_head: Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classification_head = classification_head

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs, training=training, mask=mask)
        class_x = self.head(x, training=training, mask=mask)
        x = self.decoder(x, training=training, mask=mask)
        return [x, class_x]


class GenerationAdversarialModel(Model):
    def __init__(self, encoder: Model, decoder: Model, discriminator: Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs, training=training, mask=mask)
        reconstruction_x = self.decoder(x, training=training, mask=mask)
        disc_x = self.discriminator(reconstruction_x, training=training, mask=mask)
        return [reconstruction_x, disc_x]


class ConditionalGenerationAdversarialModel(Model):
    def __init__(self, encoder: Model, decoder: Model, discriminator: Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs, training=training, mask=mask)
        reconstruction_x = self.decoder(x, training=training, mask=mask)
        disc_x = self.discriminator(tf.concat([reconstruction_x, inputs], axis=-1), training=training, mask=mask)
        return [reconstruction_x, disc_x]
