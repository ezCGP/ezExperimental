"""
Created by Michael Jurado and Yuhang Li.
This class will be how ezCGP transfers data between blocks.
"""

import Augmentor
import numpy as np


class DataSet:

    def __init__(self, x_train, y_train, x_test, y_test):
        """
        :param x_train: training samples
        :param y_train: training labels
        :param x_test: testing samples
        :param y_test: testing labels
        """
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_pipeline = Augmentor.Pipeline()
        self.test_pipeline = Augmentor.Pipeline()
        self.batch_size = None  # batch_size
        self.generator = None  # training_generator

    def clear_data(self):
        """
        Method clears the data structures so that individual can be re-evaluated
        """
        self.batch_size = None
        self.generator = None
        self.train_pipeline = Augmentor.Pipeline()
        self.test_pipeline = Augmentor.Pipeline()

    def next_batch_train(self, batch_size):
        """
        :param batch_size: mini-batch size
        :return: numpy array of training samples generated from the current training pipeline
        """
        if batch_size != self.batch_size:
            self.generator = self.train_pipeline.keras_generator_from_array(self.x_train.astype(np.uint8),
                                                                            self.y_train, batch_size, scaled=False)
        return next(self.generator)

    def preprocess_train_data(self):
        """
        Runs the augmentation and preprocessing pipeline and returns augmented and preprocessed train data
        :return: preprocessed train data (unbatched)
        """
        preprocessor = self.train_pipeline.torch_transform()
        return np.array([np.asarray(preprocessor(x)) for x in self.x_train]), self.y_train

    def preprocess_test_data(self):
        """
        Runs the preprocessing pipeline and returns preprocessed test data
        :return: preprocessed test data (unbatched)
        """
        preprocessor = self.test_pipeline.torch_transform()
        return np.array([np.asarray(preprocessor(x)) for x in self.x_test]), self.y_test
