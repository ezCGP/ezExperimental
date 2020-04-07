"""
Purpose is to have a data loader method for each dataset. Helper methods should not go here! We want this file to be
clean.
File Created by Michael Jurado, Trai Tan, and Mai Pham"""
from tensorflow.keras.utils import to_categorical
from ezData.dataset import DataSet
import numpy as np
import os
from .data_utils import split_data, load_data_from_path
from ezData.data_pair import DataPair


def load_CIFAR10(train_size_perc, validation_size_perc):
    """
    train_size_perc + validation_size_perc = 1

    train_size_perc: percentage of data in training
    validation_size_perc: percentage of data in validation set
    return
    """
    """ load all of cifar """
    path = './database/cifar-10-batches-py'
    data = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b,))
        xshape = (10000, 3, 32, 32)  # (Samples, channels, width, height)
        data.append(load_data_from_path(f, xshape))
    x = np.concatenate([x[0] for x in data])
    y = np.concatenate([x[1] for x in data])
    train, val = split_data(x, y, train_size_perc, validation_size_perc)

    x_train, y_train = train
    x_val, y_val = val

    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = to_categorical(y_train, num_classes = 10)
    x_val, y_val = np.array(x_val), np.array(y_val)
    y_val = to_categorical(y_val, num_classes = 10)
    x_train = x_train.astype(np.uint8)

    return DataSet(x_train, y_train, x_val, y_val)

def load_symbolicRegression():
    x_train = [np.float64(1), np.random.uniform(low=0.25, high=2, size=200)]
    def goal_function(data):
        return 1/data

    y_train = goal_function(x_train[1])
    return DataPair(x_train, y_train)

