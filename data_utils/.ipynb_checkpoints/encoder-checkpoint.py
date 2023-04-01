import logging
import numpy as np
import pandas as pd
from collections import defaultdict, abc
from typing import Union, Dict


UNKNOWN_LABEL = 'UNKNOWN'
PAD_LABEL = 'PAD'


class DataEncoder:

    def __init__(self, enable_padding=False, enable_unknown=False):

        self.data2index = defaultdict(int)
        self.data2count = defaultdict(int)
        self.index2data = defaultdict(str)
        self.num_of_data = 0

        if enable_padding:
            self._add_default_label(PAD_LABEL)
        if enable_unknown:
            self._add_default_label(UNKNOWN_LABEL)

    def _add_default_label(self, default_label: str):
        """Private method to add default label to encoder

        Args:
            default_label (str): the label to be added
        """
        self.index2data[self.num_of_data] = default_label
        self.data2index[default_label] = self.num_of_data
        self.data2count[default_label] = 1
        self.num_of_data += 1

    def _add_single(self, data: str):
        """Add given label to encoder

        Args:
            data (str): label to be added
        """
        if data not in self.data2index:
            self.data2index[data] = self.num_of_data
            self.index2data[self.num_of_data] = data
            self.num_of_data += 1

        self.data2count[data] += 1

    def add_data(self, data: Union[str, abc.Iterable]):
        """Method to add given data to encoder

        Args:
            data (Union[str, abc.Iterable]): data to be added to encoder.

        Raises:
            TypeError: only support str or other iterable type of data
        """
        if isinstance(data, str):
            self._add_single(data)
        elif isinstance(data, abc.Iterable):
            for d in data:
                self._add_single(d)
        else:
            raise TypeError(f'only support type of `str` and `list`, but get {type(data)}')


class CategoricalEncoder():

    LABEL_ENCODER = DataEncoder
    PAD_LABEL = PAD_LABEL
    UNKNOWN_LABEL = UNKNOWN_LABEL
    SUPPORT_MODE = ['LabelEncoding', 'VectorEncoding', 'NumericalOneHotEncoding']

    def __init__(self, col2label2idx: Dict[str, Dict[str, int]] = {},
                 col2DataEncoder: Dict[str, DataEncoder] = {}):

        self.col2label2idx = col2label2idx
        self.col2DataEncoder = col2DataEncoder

    def transform(self, list_data: pd.Series, label2idx: dict, mode='LabelEncoding') -> list:
        """Method to do categorical data transformation

        Args:
            list_data (pd.Series): input data to be transform
            label2idx (dict): label to idex mapping (ex: {'UNKNOWN': 0, '玩具公仔': 1})
            mode (str, optional): encoding mode (only support 'LabelEncoding' and 'VectorEncoding'). Defaults to 'LabelEncoding'.

        Returns:
            list: encoded features
        """

        if mode not in self.SUPPORT_MODE:
            raise ValueError(f'Only support {self.SUPPORT_MODE} mode, but get {mode}')

        encoded_features = []

        if mode == 'LabelEncoding':
            for data in list_data:
                encoded_features.append(self._encode(data, label2idx))

        elif mode == 'VectorEncoding':
            code = np.eye(len(label2idx))

            for data in list_data:
                label = self._encode(data, label2idx)

                if len(label) == 0:
                    features = [0] * len(label2idx)
                else:
                    features = np.sum(code[label], axis=0).tolist()

                encoded_features.append(features)

        elif mode == 'NumericalOneHotEncoding':
            for data in list_data:
                encoded_features.append(self._encode(data, label2idx)[0])

        return encoded_features

    def encode_transform(self, list_data: pd.Series, col: str, all_cats: dict,
                         enable_padding: bool = False, enable_unknown: bool = False, mode: str = 'LabelEncoding') -> list:
        """Method to build encoder and do categorical data transformation

        Args:
            list_data (pd.Series): input data to be transform
            col (str): target column
            all_cats (dict): all category of the given target column
            enable_padding (bool, optional): whether to enable padding token. Defaults to False.
            enable_unknown (bool, optional): whether to enable unknown token. Defaults to False.
            mode (str, optional): encoding mode (only support 'LabelEncoding' and 'VectorEncoding'). Defaults to 'LabelEncoding'.

        Returns:
            list: encoded features
        """

        # build encoder
        if not bool(all_cats):
            logging.info(f'All categories list of {col} is empty. Build from given list of data')

            if isinstance(list_data.iloc[0], str):
                all_cats = list_data.dropna().unique()
            else:
                all_cats = list_data.explode().dropna().unique()

        enc = self.build(col, all_cats, enable_padding, enable_unknown)

        # transform
        encoded_features = self.transform(list_data, self.col2label2idx[col], mode)

        return encoded_features

    def build(self, col: str, all_cats: list, enable_padding: bool = False, enable_unknown: bool = False) -> DataEncoder:
        """Method to build encoder

        Args:
            col (str): target column
            all_cats (list): all available categories of the given target column feature
            enable_padding (bool, optional): whether to enable padding token. Defaults to False.
            enable_unknown (bool, optional): whether to enable unknown token. Defaults to False.

        Returns:
            DataEncoder
        """

        enc = self.LABEL_ENCODER(enable_padding=enable_padding, enable_unknown=enable_unknown)
        enc.add_data(all_cats)
        self.col2label2idx[col] = enc.data2index
        self.col2DataEncoder[col] = enc

        return enc

    def _encode(self, data: Union[list, str], label2idx: dict) -> list:
        """Private method to get encoded features

        Args:
            data (Union[list,str]): data to be encoded
            label2idx (dict): category mapping dictionary (ex: {'玩具公仔': 0, '電玩遊戲': 1})

        Returns:
            list: encoded features
        """
        if isinstance(data, list):
            label = [label2idx.get(d, label2idx.get(self.UNKNOWN_LABEL, 0)) for d in data]
        else:
            label = [label2idx.get(data, label2idx.get(self.UNKNOWN_LABEL, 0))]

        return label

