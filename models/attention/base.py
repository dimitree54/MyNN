from models.base_classes import ModelBuilder
import tensorflow as tf


class BlockWithPostAttentionBuilder(ModelBuilder):
    def __init__(self, main_block_builder: ModelBuilder,
                 attention_block_builder: ModelBuilder):
        self.main_block_builder = main_block_builder
        self.attention_block_builder = attention_block_builder

    def build(self, filters, stride=1, **kwargs) -> tf.keras.Model:
        return tf.keras.Sequential([
            self.main_block_builder.build(filters=filters, stride=stride, **kwargs),
            self.attention_block_builder.build(**kwargs)
        ], **kwargs)
