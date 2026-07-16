"""
clustering.py - K-means 聚类
用于特征空间分组，作为辅助特征或选组依据
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_stocks(df, factor_cols=None, n_clusters=4, date_col="date"):
    """
    在每个时间截面，对股票做 K-means 聚类

    输入:
    - df: 包含因子列的 DataFrame
    - factor_cols: 用于聚类的因子列名列表
    - n_clusters: 聚类数

    返回:
    - df: 新增 'cluster' 列的 DataFrame
    - cluster_centers: 每个时间点的聚类中心（可选）
    """
    if factor_cols is None:
        factor_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]

    df = df.copy()
    df["cluster"] = np.nan

    dates = sorted(df[date_col].unique())

    for dt in dates:
        mask = df[date_col] == dt
        sub = df.loc[mask, factor_cols].dropna()

        if len(sub) < n_clusters:
            continue

        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(sub)

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        idx = sub.index
        df.loc[idx, "cluster"] = labels

    df["cluster"] = df["cluster"].astype("Int64")
    return df


def get_best_cluster(df, ret_col="fwd_ret_5d", date_col="date", cluster_col="cluster"):
    """
    计算每个聚类组的历史平均收益，选出最优簇
    返回最佳簇的标签
    """
    cluster_perf = (
        df.groupby([date_col, cluster_col])[ret_col]
        .mean()
        .reset_index()
    )
    # 每个簇在所有时间的平均收益
    avg_perf = cluster_perf.groupby(cluster_col)[ret_col].mean()
    best = avg_perf.idxmax()
    return best, avg_perf
