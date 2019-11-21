import keras
from numba import njit, prange
import numpy as np

@njit
def generate_kernels(input_length, num_kernels):

    candidate_lengths = np.array((7, 9, 11))

    # initialise kernel parameters
    weights = np.zeros((num_kernels, candidate_lengths.max())) # see note
    lengths = np.zeros(num_kernels, dtype = np.int32) # see note
    biases = np.zeros(num_kernels)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    # note: only the first *lengths[i]* values of *weights[i]* are used

    for i in range(num_kernels):

        length = np.random.choice(candidate_lengths)
        _weights = np.random.normal(0, 1, length)
        bias = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (length - 1)))
        padding = ((length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0

        weights[i, :length] = _weights - _weights.mean()
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    # zero padding
    if padding > 0:
        _input_length = len(X)
        _X = np.zeros(_input_length + (2 * padding))
        _X[padding:(padding + _input_length)] = X
        X = _X

    input_length = len(X)

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0 # "proportion of positive values"
    _max = np.NINF

    for i in range(output_length):

        _sum = bias

        for j in range(length):

            _sum += weights[j] * X[i + (j * dilation)]

        if _sum > 0:
            _ppv += 1

        if _sum > _max:
            _max = _sum

    return _ppv / output_length, _max

@njit(parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples = len(X)
    num_kernels = len(weights)

    # initialise output
    _X = np.zeros((num_examples, num_kernels * 2)) # 2 features per kernel

    for i in prange(num_examples):

        for j in range(num_kernels):

            _X[i, (j * 2):((j * 2) + 2)] = \
            apply_kernel(X[i], weights[j][:lengths[j]], lengths[j], biases[j], dilations[j], paddings[j])

    return _X

class DataGenerator(keras.utils.Sequence):

    def __init__(self, X, kernels = None,normaliser = None, batch_size=1, shuffle = False):
        self.X = X
        self.X['date'] = X.timestamp.dt.date
        self.kernels = kernels
        self.normalisers = normaliser
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.resampled_indexes = self.X[['building_id', 'date']].drop_duplicates().sample(frac=1).values.copy()
        self.on_epoch_end()
        self.len_index = int(len(self.resampled_indexes)/batch_size)

    def __len__(self):
        return self.len_index

    def __getitem__(self, index):
        # Generate indexes of the batch
        reset=True
        while reset:
            print(index)
            indexes = self.resampled_indexes[index * self.batch_size:(index + 1) * self.batch_size]
            local_X = self.X[(self.X.building_id.isin(set(indexes[:, 0]))) & (self.X.date.isin(set(indexes[:, 1])))]
            X, Y = [], []
            for local_date in local_X.groupby(['building_id', 'date']):

                X_batch = local_date[1].drop(['building_id', 'date', 'timestamp', 'log_meter_reading'], axis=1).values
                #X_batch = self.normalisers.transform(X_batch)
                Y_batch = local_date[1].log_meter_reading.values
                try:
                    X.append(X_batch.reshape((24, X_batch.shape[1],1)))
                    Y.append(Y_batch.reshape((24, 1,1)))
                except ValueError:
                    pass
                if len(X) > 0 :
                    reset = False
                else:
                    index += 1
        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        if self.shuffle is True:
            pass
            #np.random.shuffle(self.resampled_indexes)