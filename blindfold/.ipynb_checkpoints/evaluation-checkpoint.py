import numpy as np
import pandas as pd

def dcg_at_k(df_tuple, k, flag='idcg'):
    if flag == 'idcg':
        r = sorted(df_tuple, key=lambda x: 10-x[0])
    else:
        r = sorted(df_tuple, key=lambda x: x[1])
    r = [tup[0] for tup in r]
    
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k, flag='idcg')
    if not idcg:
        return 1. 
    return dcg_at_k(r, k, flag = 'dcg') / idcg

def ndcg_for_df(dataframe, pred_column, y_label, group_column, k):
    cal_df = dataframe[[group_column, y_label, pred_column]].copy()
    cal_df['tuples'] = tuple(zip(cal_df[y_label], cal_df[pred_column]))
        
    grouped_df = cal_df.groupby(group_column)['tuples'].agg(lambda x: list(x))
    grouped_df = pd.DataFrame(grouped_df).reset_index()
    grouped_df['ndcg'] = grouped_df['tuples'].apply(lambda x : ndcg_at_k(x, k))
    
    return np.mean(grouped_df['ndcg'])