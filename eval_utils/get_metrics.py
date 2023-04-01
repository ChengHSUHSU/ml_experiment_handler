from sklearn.metrics import recall_score, precision_score, ndcg_score, f1_score, roc_curve, auc
import numpy as np
import pandas as pd





def get_overall_metrics(y_true: np.array,
                        y_pred: np.array,
                        metric_type_list: list = ['recall', 'precision', 'f1', 'roc_curve', 'auc']):
    """Calculate overall metrics

    Args:
        y_true (np.array): y_true data.
        y_pred (np.array): y_pred data.
        metric_type_list (list): metric_type need to calculate metrics. Default to ['recall', 'precision', 'f1', 'roc_curve', 'auc']

    Returns:
        dict: all specific metrics result
    """
    
    y_binary = [round(value) for value in y_pred]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    result = {}
    for metric_type in metric_type_list:
        if metric_type == 'recall':
            result[metric_type] = recall_score(y_true, y_binary, average=None)
        elif metric_type == 'precision':
            result[metric_type] = precision_score(y_true, y_binary, average=None)
        elif metric_type == 'f1':
            result[metric_type] = f1_score(y_true, y_binary, average=None)
        elif metric_type == 'roc_curve':
            result[metric_type] = ({'fpr': fpr, 'tpr': tpr})
        elif metric_type == 'auc':
            result[metric_type] = auc(fpr, tpr)
    return result


def get_topk_precision_recall(df: pd.DataFrame,
                              metric_type: str = 'recall',
                              true_col: str = 'y_true',
                              pred_col: str = 'y_pred',
                              sort_by: list = ['y_true', 'y_pred'],
                              k: int = 5):
    """Function to get topk Precision-Recall score

    Args:
        df (pd.DataFrame)
        metric_type (str, optional): parameter to select metric type, support `recall` and `precision`. Defaults to 'recall'.
        true_col (str, optional): ground true column name. Defaults to 'y_true'.
        pred_col (str, optional): predict column name. Defaults to 'y_pred'.
        sort_by (list, optional): Defaults to ['y_true', 'y_pred'].
        k (int, optional): variable to calculate top k score. Defaults to 5.

    Returns:
        list
    """

    if metric_type not in ['recall', 'precision']:
        raise Exception('Function only support metric `recall` and `precision`')

    df = df.sort_values(by=sort_by, ascending=False)[:k]
    data_pred = np.round(df[pred_col])
    data_true = df[true_col]

    if metric_type == 'recall':
        result = recall_score(data_pred, data_true, average=None)
    elif metric_type == 'precision':
        result = precision_score(data_pred, data_true, average=None)

    return result


def get_average_metrics(df: pd.DataFrame,
                        metric_type: str = 'ndcg',
                        groupby_key='openid',
                        true_col: str = 'y_true',
                        pred_col: str = 'y_pred',
                        sort_by: list = ['y_true', 'y_pred'],
                        k: int = 5):
    """Function to get average metric score

    Args:
        df (pd.DataFrame)
        metric_type (str, optional): parameter to select metric type, support `ndcg`, `recall` and `precision`. Defaults to 'ndcg'.
        groupby_key (str, optional): parameter to select which column to group by. Defaults to 'openid'.
        true_col (str, optional): ground true column name. Defaults to 'y_true'.
        pred_col (str, optional): predict column name. Defaults to 'y_pred'.
        sort_by (list, optional): Defaults to ['y_true', 'y_pred'].
        k (int, optional): variable to calculate top k score. Defaults to 5.

    Returns:
        float (ndcg)
        list (recall, precision)
    """

    if metric_type not in ['ndcg', 'recall', 'precision']:
        raise Exception('Function only support metric `ndcg`, `recall` and `precision`')

    df_true = df.groupby(groupby_key)[true_col].apply(list).reset_index(name=f'{true_col}_list')
    df_pred = df.groupby(groupby_key)[pred_col].apply(list).reset_index(name=f'{pred_col}_list')
    df = df_true.merge(df_pred, on=groupby_key, how='left')

    score = 0

    if metric_type == 'ndcg':
        for _, row in df.iterrows():
            if(len(row[f'{true_col}_list']) > 1):
                score += ndcg_score(np.array([row[f'{true_col}_list']]),
                                    np.array([row[f'{pred_col}_list']]),
                                    k=k)
    elif metric_type in ['recall', 'precision']:
        for _, row in df.iterrows():
            df_pair = pd.DataFrame(data={true_col: np.array(row[f'{true_col}_list']),
                                         pred_col: np.array(row[f'{pred_col}_list'])})

            score += get_topk_precision_recall(df_pair, metric_type=metric_type, k=k, sort_by=sort_by)

    return score/len(df)
