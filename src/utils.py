"""
utils.py - 工具函数：日期处理、标准化、评价指标
"""

import numpy as np
import pandas as pd

def standardize(df, method="zscore"):
    """
    因子标准化
    method: 'zscore' | 'rank' | 'mad' (MAD截尾)
    """
    df = df.copy()
    if method == "zscore":
        df = (df - df.mean()) / df.std()
    elif method == "rank":
        df = df.rank(pct=True)
    elif method == "mad":
        median = df.median()
        mad = (df - median).abs().median()
        df = (df - median) / (1.4826 * mad + 1e-8)
        df = df.clip(-3, 3)
    return df


def neutralize(df, factors):
    """对因子做行业/风格中性化（回归残差）"""
    from sklearn.linear_model import LinearRegression
    y = df.values
    X = factors.values
    model = LinearRegression().fit(X, y)
    residual = y - model.predict(X)
    return pd.Series(residual.flatten(), index=df.index)


def winsorize(series, limits=(0.01, 0.01)):
    """去极值：分位数截尾"""
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower, upper)


def _pearsonr(x, y):
    """纯 numpy 实现的皮尔逊相关系数"""
    xm = x - x.mean()
    ym = y - y.mean()
    r = np.sum(xm * ym) / (np.sqrt(np.sum(xm ** 2)) * np.sqrt(np.sum(ym ** 2)) + 1e-10)
    return r


def _spearmanr(x, y):
    """纯 numpy 实现的斯皮尔曼秩相关系数"""
    x_rank = np.argsort(np.argsort(x)).astype(float)
    y_rank = np.argsort(np.argsort(y)).astype(float)
    return _pearsonr(x_rank, y_rank)


def calc_ic(factor_series, forward_return_series, method="spearman"):
    """
    计算截面IC

    Parameters:
    - factor_series: 当前期因子值
    - forward_return_series: 未来N日收益
    - method: 'pearson' | 'spearman'

    Returns:
    - ic_value: float
    """
    df = pd.DataFrame({
        "factor": factor_series,
        "ret": forward_return_series
    }).dropna()
    if len(df) < 10:
        return np.nan
    x = df["factor"].values
    y = df["ret"].values
    if method == "spearman":
        return _spearmanr(x, y)
    else:
        return _pearsonr(x, y)


def calc_ic_ir(ic_series):
    """IC_IR = mean(IC) / std(IC)"""
    return ic_series.mean() / (ic_series.std() + 1e-8)


def calc_sharpe(returns, rf=0.0, periods=252):
    """计算年化夏普比率"""
    excess = returns.values if isinstance(returns, pd.Series) else returns
    excess = excess - rf / periods
    if np.std(excess) == 0:
        return 0.0
    return np.sqrt(periods) * np.mean(excess) / np.std(excess)


def calc_max_drawdown(nav):
    """计算最大回撤"""
    peak = nav.expanding().max()
    dd = (nav - peak) / peak
    return dd.min()


def calc_annual_return(nav, periods_per_year=252):
    """计算年化收益率"""
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(nav) / periods_per_year
    return (1 + total_ret) ** (1 / n_years) - 1
