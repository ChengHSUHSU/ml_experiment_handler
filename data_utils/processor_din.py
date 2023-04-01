import json
import numpy as np
import pandas as pd

#from bdds_recommendation.src.preprocess.utils.din import get_topk_preference_data, padding


def get_behavior_feature(dataset: pd.DataFrame,
                         encoder: dict,
                         encoder_key: str = 'cat1',
                         behavior_col: str = 'user_category',
                         seq_length_col: str = 'seq_length',
                         prefix: str = 'hist_',
                         profile_key: str = 'click',
                         sequence_key: str = 'cat1',
                         max_sequence_length: int = 10):
    """Function to get DIN behavior feature from preference profile

    Args:
        dataset (pd.DataFrame): pandas dataframe
        encoder (dict): feature encoder
        encoder_key (str, optional): encoder key. Defaults to 'cat1'.
        behavior_col (str, optional): target column for behavior sequence. Defaults to 'user_category'.
        seq_length_col (str, optional): sequence length column name. Defaults to 'seq_length'.
        prefix (str, optional): behavior sequence column prefix. Defaults to 'hist_'.
        profile_key (str, optional): preference profile key. Defaults to 'click'.
        sequence_key (str, optional): preference sequence key. Defaults to 'cat1'.
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.

    Returns:
        (training) pandas dataframe
        (serving) list
    """

    pad_token = encoder.get('PAD', None)
    unknown_token = encoder.get('UNKNOWN', None)

    if pad_token is None:
        raise KeyError('The PAD_LABEL need to be set in encoder to generate sequence data')

    if unknown_token is None:
        raise KeyError('The UNKNOWN_LABEL need to be set in encoder to generate sequence data')

    hist_col_name = f'{prefix}{encoder_key}'

    dataset[hist_col_name] = dataset[behavior_col].apply(lambda x: get_topk_preference_data(x,
                                                                                            profile_key=profile_key,
                                                                                            sequence_key=sequence_key,
                                                                                            max_sequence_length=max_sequence_length))

    dataset[seq_length_col] = dataset[hist_col_name].apply(len)
    dataset[hist_col_name] = dataset[hist_col_name].apply(lambda hist_list: [encoder.get(val, unknown_token) for val in hist_list])
    dataset[hist_col_name] = dataset[hist_col_name].apply(lambda hist_list: padding(hist_list, max_sequence_length, pad_token=pad_token))

    return dataset




def get_topk_preference_data(data: str, profile_key: str = 'click', sequence_key: str = 'cat1', max_sequence_length: int = 10):
    """Function to parse top k preference data from user profile.
    Support user profiles with the format of {<profile_key>: <sequence_key>: {id_1: {pref: xxx}}}

    Args:
        data (str)
        profile_key (str, optional): preference profile key. Defaults to 'click'.
        sequence_key (str, optional): preference sequence key. Defaults to 'cat1'.
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.

    Returns:
        list
    """
    try:
        result = json.loads(data).get(profile_key, {}).get(sequence_key, [])
    except json.JSONDecodeError:
        result = []

    if not result:
        return result

    key2pref = {k: v['pref'] if 'pref' in v else 0 for k, v in result.items()}
    key2pref = sorted(key2pref.items(), key=lambda kv: kv[1], reverse=True)
    sequence_length = len(key2pref)

    max_sequence_length = sequence_length if sequence_length < max_sequence_length else max_sequence_length
    result = [key2pref[i][0] for i in range(max_sequence_length)]
    return result


def padding(data: list, max_sequence_length: int = 10, pad_token: int = 0):
    """Function for padding list to `max_sequence_length` with `pad_token`

    Args:
        data (list)
        max_sequence_length (int, optional): maximum sequence length. Defaults to 10.
        pad_token (int, optional): pad_token. Defaults to 0.

    Returns:
        np.array
    """

    if max_sequence_length < len(data):
        return np.array(data[:max_sequence_length])

    data = np.pad(np.array(data),
                  pad_width=[0, max_sequence_length-len(data)],
                  mode='constant',
                  constant_values=pad_token)
    return data

