"""
factors.py - 因子计算、处理、IC分析
"""

import numpy as np
import pandas as pd
from src.utils import winsorize, standardize, calc_ic, calc_ic_ir


def calc_factors(df):
    """
    计算三个核心因子：
    - PE (TTM)
    - 20日动量 Momentum_20
    - 60日波动率 Vol_60

    输入 df 需包含列：date, code, close, pe
    返回：df 新增因子列
    """
    df = df.copy()
    df = df.sort_values(["code", "date"])

    # 1. PE 因子（直接用日频 PE，实际中应使用截面数据时点 PE）
    df["factor_pe"] = df["pe"]

    # 2. 20日动量 = (close / close.shift(20)) - 1
    df["factor_momentum_20"] = (
        df.groupby("code")["close"].transform(lambda x: x / x.shift(20) - 1)
    )

    # 3. 60日波动率 = 过去60日收益率的标准差
    df["daily_ret"] = df.groupby("code")["close"].transform(lambda x: x.pct_change())
    df["factor_vol_60"] = (
        df.groupby("code")["daily_ret"]
        .transform(lambda x: x.rolling(60, min_periods=20).std())
    )

    return df


def process_factors(df, factor_cols=None, winsor_limits=(0.01, 0.01), std_method="mad"):
    """
    因子预处理：去极值 → 标准化
    """
    if factor_cols is None:
        factor_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]

    for col in factor_cols:
        df[col] = df.groupby("date")[col].transform(
            lambda x: winsorize(x, limits=winsor_limits)
        )
        df[col] = df.groupby("date")[col].transform(
            lambda x: standardize(x, method=std_method)
        )
    return df


def calc_ic_analysis(df, factor_cols=None, ret_col="fwd_ret_5d", method="spearman"):
    """
    计算每个因子在每个 time step 的截面 IC
    以及汇总：IC mean, IC std, IC_IR, IC>0比率

    返回:
    - ic_df: 每期每因子的 IC (date × factor)
    - ic_summary: 汇总统计
    """
    if factor_cols is None:
        factor_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]

    records = []
    dates = sorted(df["date"].unique())

    for dt in dates:
        day_data = df[df["date"] == dt].dropna(subset=[ret_col] + factor_cols)
        if len(day_data) < 10:
            continue
        for factor in factor_cols:
            ic = calc_ic(day_data[factor], day_data[ret_col], method=method)
            records.append({"date": dt, "factor": factor, "IC": ic})

    ic_df = pd.DataFrame(records)

    if ic_df.empty:
        print("[WARN] IC 计算无有效数据")
        return ic_df, pd.DataFrame()

    # 汇总统计
    summary = (
        ic_df.groupby("factor")["IC"]
        .agg(["mean", "std", lambda x: calc_ic_ir(x), lambda x: (x > 0).mean()])
        .rename(columns={
            "mean": "IC_mean",
            "std": "IC_std",
            "<lambda_0>": "IC_IR",
            "<lambda_1>": "IC_positive_ratio",
        })
        .reset_index()
    )

    # 年度统计
    ic_df["year"] = ic_df["date"].dt.year
    yearly = (
        ic_df.groupby(["factor", "year"])["IC"]
        .agg(["mean", "std"])
        .reset_index()
    )

    return ic_df, summary, yearly
