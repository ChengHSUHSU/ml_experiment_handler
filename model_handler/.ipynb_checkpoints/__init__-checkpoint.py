# -*- coding: utf-8 -*-
from abc import *


class BaseModelHandler(metaclass=ABCMeta):
    """Model Handler for machine learning module"""

    MODE = ['train', 'validation', 'inference']

    def __init__(self, mode='train'):

        assert mode in self.MODE, 'only support train, validation and inference mode'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def save_model(self, output_path):
        pass

    def set_input(self, input_data):
        return input_data
