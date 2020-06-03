import glob
import os
from typing import Any

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging


class ModelCheckpointBestAndLast(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 filepath: Any,
                 monitor: str = 'val_loss',
                 verbose: int = 0,
                 mode: str = 'auto',
                 **kwargs: Any):
        super().__init__(filepath, monitor, verbose, False, False, mode, 'epoch', **kwargs)
        self.prev_path = None
        self.best_path = None

    @staticmethod
    def _remove_prev(path):
        if path:
            os.remove(path + ".index")
            file_list = glob.glob(path + ".data*")
            for data_path in file_list:
                os.remove(data_path)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        filepath = self._get_file_path(epoch, logs)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
        else:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s' % (epoch + 1, self.monitor,
                                                   self.best, current, filepath))
                self._remove_prev(self.best_path)
                self.best = current
                self.best_path = filepath
            else:
                self._remove_prev(self.prev_path)
                self.prev_path = filepath
