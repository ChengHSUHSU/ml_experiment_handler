import torch
import pandas as pd
import numpy as np
from typing import Union

from model_handler import BaseModelHandler
from model_handler.utils.features import get_sparse_feature_columns, get_dense_feature_columns, \
    get_behavior_feature_columns
from model_handler.utils.inputs import get_feature_names
from model_handler.model.din import DIN


class DINHandler(BaseModelHandler):

    def __init__(self, col2label2idx: dict, config: dict, mode: str = 'train', use_cuda: bool = False, **kwargs):

        super(DINHandler, self).__init__(mode)

        self.device = 'cuda:0' if (use_cuda and torch.cuda.is_available()) else 'cpu'
        self.config = config
        self.col2label2idx = col2label2idx
        self.mode = mode

        kwargs['device'] = self.device
        self.initialize(**kwargs)

    def code(cls):
        return 'DIN'

    def initialize(self, **kwargs):

        self.model = DIN(dnn_feature_columns=self.get_feature_columns(),
                         history_feature_list=self.config['BEHAVIOR_FEATURE'],
                         **kwargs)

        if self.mode == 'train':
            self.model.compile(optimizer=kwargs.get('optimizer', 'adagrad'),
                               loss=kwargs.get('objective_function', 'binary_crossentropy'),
                               metrics=kwargs.get('metrics', ['binary_crossentropy']))

    def train(self, x_train: pd.DataFrame, y_train: Union[pd.Series, np.array, list], train_params: dict):

        x_train = self.set_input(x_train)
        y_train = np.array(y_train)

        result = self.model.fit(x_train, y_train, **train_params)
        return result

    def predict(self, df: pd.DataFrame):
        df = self.set_input(df)

        return self.model.predict(df)

    def load_model(self, model_path: str):

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def save_model(self, output_path: str):

        torch.save(self.model.state_dict(), output_path)

    def set_input(self, df: pd.DataFrame):

        feature_dict = {}

        for column_name in df:
            feature_dict[column_name] = np.array(df[column_name].to_list())

        input_features = {name: feature_dict[name] for name in get_feature_names(self.feature_columns)}
        return input_features

    def get_feature_columns(self) -> list:
        """Public method to get feature columns

        Returns:
            list: list of feature columns
        """
        sparse_feat_cols = get_sparse_feature_columns(sparse_feature_cols=self.config['ONE_HOT_FEATURE'],
                                                      feature_embed_size_map=self.config['FEATURE_EMBEDDING_SIZE'],
                                                      col2label2idx=self.col2label2idx)

        dense_feat_cols = get_dense_feature_columns(dense_feature_cols=self.config['DENSE_FEATURE'],
                                                    dense_embed_size_map=self.config['DENSE_FEATURE_SIZE'])

        behavior_feat_cols = get_behavior_feature_columns(behavior_feature_cols=self.config['BEHAVIOR_FEATURE'],  
                                                          behavior_feature_len_cols=self.config['BEHAVIOR_FEATURE_SEQ_LENGTH'],
                                                          feature_embed_size_map=self.config['FEATURE_EMBEDDING_SIZE'], 
                                                          col2label2idx=self.col2label2idx,
                                                          behavior_seq_size=self.config['BEHAVIOR_SEQUENCE_SIZE'])

        self.feature_columns = sparse_feat_cols + dense_feat_cols + behavior_feat_cols

        return self.feature_columns
