import os
import numpy as np
import numpy as np
# from scipy.stats import weibull_min
from tensorflow.keras.utils import to_categorical
import os
import sys
import random
import six
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from copy import deepcopy

from .db_config import DbConfig
from .dataset import DataSet


class DbManager:
    def __init__(self, config: DbConfig):
        # self.train_data_set: DataSet = None
        # self.test_data_set: DataSet = None
        # self.db_conf: DbConfig = config
        self.train_data_set = None
        self.test_data_set = None
        self.db_conf = config
        pass

    def load_data_from_path(self, path):
        with open(path, 'rb') as f:
            if six.PY2:
                datadict = pickle.load(f)
            elif six.PY3:
                datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def split_data(self, X, y):
        assert self.db_conf.train_size_perc + self.db_conf.validation_size_perc == 1

        train_index = int(len(X) * self.db_conf.train_size_perc)
        X_train = X[0:train_index]
        y_train = y[0:train_index]

        X_val = X[train_index:]
        y_val = y[train_index:]

        return (X_train, y_train), (X_val, y_val)

    def get_train_data(self):
        return self.train_data_set

    def get_test_data(self):
        return self.test_data_set

    def clone_train_data(self):
        return deepcopy(self.train_data_set)

    def clone_test_data(self):
        return deepcopy(self.test_data_set)

    def load_CIFAR10(self):
        """ load all of cifar """
        path = './database/cifar-10-batches-py'
        data = []
        for b in range(1, 6):
            f = os.path.join(path, 'data_batch_%d' % (b,))
            data.append(self.load_data_from_path(f))
        x = np.concatenate([x[0] for x in data])
        y = np.concatenate([x[1] for x in data])
        train, val = self.split_data(x, y)

        test_path = os.path.join(path, 'test_batch')
        test = self.load_data_from_path(test_path)

        x_train, y_train = train
        # x_test, y_test = test
        x_val, y_val = val

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = to_categorical(y_train, num_classes = 10)
        # x_test, y_test = np.array([x_test]), np.array([y_test])
        x_val, y_val = np.array(x_val), np.array(y_val)
        y_val = to_categorical(y_val, num_classes = 10)
        x_train = x_train.astype(np.uint8)

        return DataSet(x_train, y_train, x_val, y_val)