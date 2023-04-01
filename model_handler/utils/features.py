from typing import List
from model_handler.utils.inputs import DenseFeat, SparseFeat, VarLenSparseFeat


def get_sparse_feature_columns(sparse_feature_cols: list, feature_embed_size_map: dict, col2label2idx: dict) -> List[SparseFeat]:
    """Function to get sparse feature like age, gender, cat, etc

    Args:
        sparse_feature_cols (list): sparse feature column names, ex: ['age', 'gender', 'cat0', 'cat1']
        feature_embed_size_map (dict): embedding size mapping for sparse feature columns, ex: {'age': 8, 'gender': 2, 'cat0': 17, 'cat1': 100}
        col2label2idx (dict): columns to labels mapping for calculating sparse feature length

    Returns:
        List[SparseFeat]: sparse features in SparseFeat type for deep ctr models
    """

    sparse_feature_columns = []

    for feature_column_name in sparse_feature_cols:

        feature_length = len(col2label2idx[feature_column_name])

        embedding_size = feature_embed_size_map[feature_column_name]
        sparse_feature_columns.append(
            SparseFeat(
                feature_column_name,
                feature_length,
                embedding_dim=embedding_size
            ))
    return sparse_feature_columns


def get_dense_feature_columns(dense_feature_cols: list, dense_embed_size_map: dict, default_embed_size: int = 1) -> List[DenseFeat]:
    """Function to get dense feature like final_score, pref_score, user_title_embedding, item_title_embedding etc

    Args:
        dense_feature_map (list): dense feature cols
                                  ex: ['final_score', 'cat0_pref_score', 'user_title_embedding', 'item_title_embedding]
        dense_embed_size_map (dict): dense feature col and embed size mapping
                                     ex: {'final_score': 1, 'cat0_pref_score': 1, 'user_title_embedding': 300, 'item_title_embedding': 300}
        default_embed_size (int): default dense feature embedding size. default to 1

    Returns:
        List[DenseFeat]: dense features in DenseFeat type for deep ctr models
    """

    dense_feature_columns = []

    for feature_column_name in dense_feature_cols:
        embedding_size = dense_embed_size_map.get(feature_column_name, default_embed_size)
        dense_feature_columns += [DenseFeat(feature_column_name, embedding_size)]

    return dense_feature_columns


def get_behavior_feature_columns(behavior_feature_cols: list, behavior_feature_len_cols: list,
                                 feature_embed_size_map: dict, col2label2idx: dict,
                                 behavior_seq_size: int = 10) -> List[VarLenSparseFeat]:
    """Function to get behavior feature like hist_cat, hist_tag, etc

    Args:
        behavior_feature_cols (list): behavior feature column names, ex: ['cat0', 'cat1']
        behavior_feature_len_cols (list): behavior sequence length column names, ex: ['seq_length_cat0', 'seq_length_cat1']
        feature_embed_size_map (dict): embedding size mapping for sparse feature columns, ex: {'age': 8, 'gender': 2, 'cat0': 17, 'cat1': 100}
        col2label2idx (dict): columns to labels mapping for calculating sparse feature length
        behavior_seq_size (int): behavior sequence length. Defaults to 10.

    Returns:
        List[VarLenSparseFeat]: behavior features in VarLenSparseFeat type for DIN models
    """

    behavior_feature_columns = []

    for (feature_column_name, feature_seq_len_name) in zip(behavior_feature_cols, behavior_feature_len_cols):
        embedding_size = feature_embed_size_map[feature_column_name]

        feature_length = len(col2label2idx[feature_column_name])

        behavior_feature_columns += [
            VarLenSparseFeat(
                SparseFeat(
                    f'hist_{feature_column_name}',
                    feature_length + 1,  # TODO: remove +1 along with serving code
                    embedding_dim=embedding_size),
                behavior_seq_size,
                length_name=feature_seq_len_name
            )
        ]

    return behavior_feature_columns
