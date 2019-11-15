import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, Y,time_series_len = 5, batch_size=200, shuffle=True):
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()
        self.len_index = range(len(X))


    def __len__(self):
        return int(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.resampled_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.X[indexes], self.Y[indexes]

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.resampled_indexes)