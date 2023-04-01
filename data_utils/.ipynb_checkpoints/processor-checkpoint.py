import ast
import json
import pandas as pd
from data_utils.load import read_pickle
from data_utils.encoder import CategoricalEncoder 
import swifter
from data_utils.processor_din import get_behavior_feature

    
def convert_columns_name(dataset, configs):
    print('starting convert_columns_name...')
    dataset = dataset.rename(columns=configs['COLUMN_TO_RENAME'])
    return dataset



def handle_missing_data(dataset, configs):
    print('starting handle_missing_data...')
    for key, fillna_value in configs['COLUMN_TO_FILLNA'].items():
        if isinstance(fillna_value, list):
            dataset[key] = dataset[key].swifter.apply(lambda x: x if x else fillna_value)
        elif callable(fillna_value):
            dataset[key] = dataset[key].swifter.apply(fillna_value)
        else:
            dataset[key] = dataset[key].fillna(fillna_value)

    return dataset


def checknull(x):

    result = False

    if (x is None) or (not bool(x)):
        result = True
    elif not isinstance(x, str):
        if (math.isnan(x)):
            result = True
        elif (np.isnan(x)):
            result = True

    return result

def convert_data_type(dataset, configs, mode='train', verbose=0):
    print('starting convert_data_type...')
    if mode == 'inference':
        dataset = convert_type(dataset, configs['TYPE_CONVERT_MODE2COLS_INFERENCE'], verbose=verbose)
    else:
        dataset = convert_type(dataset, configs['TYPE_CONVERT_MODE2COLS'], verbose=verbose)
    return dataset


def convert_type(dataset, mode2cols, verbose=True):
    """Function to convert col values by ast.literal_eval or json.loads (ex: embedding: '[1,2,....,0]' -> [1,2,...,0])
    """
    def _ast_convert_type(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    def _json_convert_type(x):
        # json.loads is faster for embedding
        if isinstance(x, str):
            return json.loads(x)
        return x

    for mode, cols in mode2cols.items():
        for col in cols:
            if mode == 'ast':
                dataset[col] = dataset[col].swifter.apply(_ast_convert_type)
            elif mode == 'json':
                dataset[col] = dataset[col].swifter.apply(_json_convert_type)
            else:
                dataset[col] = dataset[col].astype(mode)
    return dataset



def process_age(df, columns=['age'], age_range=20, UNKNOWN_LABEL='unknown'):
    """Function to convert age from numerical data to categorical data
    Args:
        df (pandas.Dataframe): dataframe
        columns (list, optional): list of age column names. Defaults to 'age'.
        age_range (int, optional): range of age group. Defaults to 20.

    Returns:
        df (pandas.Dataframe)
    """
    print('starting process_age...')
    columns = [columns] if isinstance(columns, str) else columns

    for column_name in columns:
        if column_name in df.columns:
            df[column_name] = df[column_name].swifter.apply(lambda x: str(int(x)//age_range) if not bool(checknull(x)) else UNKNOWN_LABEL)

    return df


def process_category(dataset, configs):
    print('starting process_category...')
    for col in configs['CATEGORY_COLS']:
        dataset[col] = dataset[col].swifter.apply(lambda x: [x[0]] if len(x) >= 1 else ['unknown'])
    return dataset


def aggregate_preference(dataset, configs, mode='train'):
    print('---start aggregate_preference---')
    user_pref_helper = Preference(configs)

    for cat_pref_score_paras in configs['CATEGORY_PREF_SCORE_PROCESS']:
        dataset = user_pref_helper.extract_category_pref_score(df=dataset,
                                                               enable_single_user=(mode == 'inference'),
                                                               **cat_pref_score_paras)

    for tag_pref_score_paras in configs['TAG_PREF_SCORE_PROCESS']:
        dataset = user_pref_helper.extract_tag_pref_score(df=dataset,
                                                          enable_single_user=(mode == 'inference'),
                                                          **tag_pref_score_paras)
    return dataset


class Preference():

    def __init__(self, config):
        self.config = config

    def _calculate_user_pref_score(self, user_data, item_data):
        """Dot product calculation of user data and item_data"""
        result = 0.0
        if isinstance(item_data, str):
            result = user_data.get(item_data, 0.0)      # One-hot data
        elif isinstance(item_data, list):
            for val in item_data:
                result += user_data.get(val, 0.0)       # Multi-hot data
        return result

    def _parse_tags_from_string(self, tags):
        """Parse tags from string to list, and extract text value"""
        result = []
        if isinstance(tags, str):
            tags = ast.literal_eval(tags)
        for val in tags:
            if '-' in val and len(val) > 2:
                result.append(val[2:])
            else:
                result.append(val)
        return result

    def _get_pref_scores_mapping(self, x, level=['click', 'cat1']):
        """Parse user category profile from string to dict, and extract key-value pair of preference score"""
        result = {}

        x = json.loads(x).get(level[0], {}) \
                         .get(level[1], {})

        for key, val in x.items():
            result[self._remove_entity_prefix(key)] = val.get('pref', 0.0)
        return result

    def _remove_entity_prefix(self, text):
        return text[text.find('-') + 1:]

    def extract_category_pref_score(self, df, level=['click', 'cat1'], cat_col='user_category', score_col='category_pref_score', enable_single_user=False):
        # Preprocess of raw category profile parsing

        if enable_single_user:
            """
            enable_single_user is used for single user ranking, performance increase: O(N^2) -> O(N)
            Model training: `enable_single_user=False`
            Online serving: `enable_single_user=True`
            """
            user_cat_dict = self._get_pref_scores_mapping(df[cat_col].iloc[0], level=level)
            df[score_col] = df.swifter.apply(lambda x: self._calculate_user_pref_score(user_cat_dict, x[level[1]]), axis=1)
        else:
            df['parsed_user_cat'] = df[cat_col].apply(lambda x: self._get_pref_scores_mapping(x, level=level) if pd.notna(x) else {})
            df[score_col] = df.swifter.apply(lambda x: self._calculate_user_pref_score(x['parsed_user_cat'], x[level[1]]), axis=1)

        return df

    def extract_tag_pref_score(self, df, tag_entity_list=[], user_tag_col='user_tag', item_tag_col='tags', tagging_type='editor', score_col='', enable_single_user=False):

        for tag_key in tag_entity_list:
            tagging_column_name = score_col if score_col else f'user_tag_{tagging_type}_{tag_key}'

            if enable_single_user:
                """
                enable_single_user is used for single user ranking, performance increase: O(N^2) -> O(N)
                Model training: `enable_single_user=False`
                Online serving: `enable_single_user=True`
                """
                user_tagging_dict = self._get_pref_scores_mapping(df[user_tag_col].iloc[0], level=[tagging_type, tagging_column_name])
                df[tagging_column_name] = df.swifter.apply(lambda x: self._calculate_user_pref_score(user_tagging_dict, x[item_tag_col]), axis=1)
            else:
                df[tagging_column_name] = df[user_tag_col].swifter.apply(lambda x: self._get_pref_scores_mapping(x, level=[
                                                                 tagging_type, tag_key]) if pd.notna(x) else {})
                df[tagging_column_name] = df.swifter.apply(lambda x: self._calculate_user_pref_score(x[tagging_column_name], x[item_tag_col]), axis=1)
        return df



    
def encode_features(dataset, configs, train_params, col2label2idx={}, prefix='', suffix='', mode='train'):
    print('---start encode_features---')
    data_encoder = CategoricalEncoder(col2label2idx=col2label2idx)

    if mode == 'train' and not col2label2idx:
        if train_params.get('all_cats'):
            all_cats = train_params['all_cats']
        else:
            all_cats_file = configs['COL2CATS_NAMES']
            all_cats = read_pickle(all_cats_file, base_path=train_params.get('dataroot', './dataset'))
            print('all_cats : ', all_cats.keys())

        for feature_col, (enable_padding, enable_unknown, mode) in configs['CATEGORY_FEATURES_PROCESS'].items():
            col = feature_col.replace(prefix, '').replace(suffix, '')
            dataset[feature_col] = data_encoder.encode_transform(dataset[feature_col], feature_col, all_cats[col],
                                                                 enable_padding, enable_unknown, mode)
    else:
        for feature_col, (_, _, mode) in configs['CATEGORY_FEATURES_PROCESS'].items():
            col = feature_col.replace(prefix, '').replace(suffix, '')
            dataset[feature_col] = data_encoder.transform(dataset[feature_col], col2label2idx[col], mode)

    encoder = data_encoder
    col2label2idx_new = encoder.col2label2idx
    return dataset, col2label2idx_new
    
    

def process_time_period(df, start_time_col='publish_time', end_time_col='timestamp', time_type=['min', 'hour'], postfix='_time_period'):
    """Function to compute time period, input `start_time` and `end_time` must be second.

    Args:
        df (pandas.Dataframe): dataframe
        start_time_col (str, optional): start time column name. Defaults to 'publish_time'.
        end_time_col (str, optional): end time column name. Defaults to 'timestamp'.
        time_type (list, optional): time period type. Defaults to ['min', 'hour'].
        postfix (str, optional): postfix of time period column. Defaults to '_time_period'.

    Returns:
        df (pandas.Dataframe)
    """
    print('---start process_time_period---')

    time_type = [time_type] if isinstance(time_type, str) else time_type

    time_type_to_sec = {'min': 60, 'hour': 3600}

    for t in time_type:

        df[f'{t}{postfix}'] = df.swifter.apply(lambda x: (x[end_time_col]-x[start_time_col]) / (time_type_to_sec[t]), axis=1)

    return df


def process_normalize_data(dataset, configs):
    print('---start process_normalize_data---')
    dataset = normalize_data(dataset, configs['NORMALIZE_COLS'])
    return dataset


def normalize_data(dataset, cols):
    for mode in cols:
        if mode not in ['min-max', 'z-score']:
            raise ValueError(f'Only support [`min-max`, `z-score`] mode, but get {mode}')

        if mode == 'min-max':
            for col in cols[mode]:
                max = cols[mode][col]
                min = dataset[col].min()
                if max is not None:  # upper bound value
                    dataset[col] = dataset[col].swifter.apply(lambda x: 1 if calculate_min_max(x, max, min) > 1 else calculate_min_max(x, max, min))
                else:
                    dataset[col] = dataset[col].swifter.apply(lambda x: calculate_min_max(x, dataset[col].max(), min))

        if mode == 'z-score':
            for col in cols[mode]:
                dataset[col] = dataset[col].swifter.apply(lambda x: (x - dataset[col].mean()) / dataset[col].std())

    return dataset


def calculate_min_max(x, max_value, min_value):
    '''Function to do min-max normalization to input value

    :param x: input value
    :type x: float
    :param max_value: maximum value
    :type max_value: float
    :param min_value: minimum value
    :type min_value: float
    :return: normalized value
    :rtype: float
    '''
    normed_value = (x - min_value) / (max_value - min_value) if max_value - min_value > 0 else 0

    return normed_value



def process_user_behavior_sequence(dataset, configs, col2label2idx_new):
    print('---start process_user_behavior_sequence---')
    max_seq_len = configs['BEHAVIOR_SEQUENCE_SIZE']

    for col, list_params in configs['BEHAVIOR_SEQUENCE_FEATURES_PROCESS'].items():
        for params in list_params:
            encoder_key, event_col, level, hist_suffix, seq_len_suffix = params

            dataset = get_behavior_feature(dataset=dataset,
                                           encoder=col2label2idx_new[encoder_key],
                                           behavior_col=col,
                                           encoder_key=encoder_key,
                                           seq_length_col=f'{seq_len_suffix}{encoder_key}',
                                           prefix=hist_suffix,
                                           profile_key=event_col,
                                           sequence_key=level,
                                           max_sequence_length=max_seq_len)
    return dataset

