
"""
data_loader.py - 数据加载和预处理
支持两种模式：
1. 从本地 CSV 加载（需用户自行下载）
2. 生成模拟数据（用于测试项目流程）
"""

import numpy as np
import pandas as pd


def filter_stocks(df, min_pct=0.60):
    """
    过滤数据不足的股票

    参数:
    - df: 日线 DataFrame，需包含 date, code 列
    - min_pct: 最小交易日覆盖率（默认60%）

    返回:
    - 过滤后的 DataFrame + 被过滤的股票列表
    """
    total_trading_days = df["date"].nunique()
    threshold = int(total_trading_days * min_pct)
    counts = df.groupby("code").size()
    good_codes = counts[counts >= threshold].index
    bad_codes = counts[counts < threshold].index

    before = len(df)
    df = df[df["code"].isin(good_codes)].copy()
    after = len(df)

    print(f"  [清洗] 交易日: {total_trading_days}, 阈值: {threshold}行 ({min_pct*100:.0f}%)")
    print(f"  [清洗] 过滤前: {len(counts)} 只")
    print(f"  [清洗] 过滤掉: {len(bad_codes)} 只 (数据量不足)")
    print(f"  [清洗] 保留: {len(good_codes)} 只")
    print(f"  [清洗] {before} 行 → {after} 行")

    return df, bad_codes.tolist()


def load_data(filepath=None, start_date="2018-01-01", end_date="2022-12-31",
              filter_min_pct=0.60):
    """
    加载沪深300成分股日线数据

    参数:
    - filepath: CSV 文件路径（None 则生成模拟数据）
    - filter_min_pct: 股票过滤阈值（0-1），设为 0 不过滤

    期望CSV格式：
        date, code, pe, close, volume, industry
    """
    if filepath:
        df = pd.read_csv(filepath, parse_dates=["date"])
    else:
        print("[INFO] 未提供数据文件，使用模拟数据演示流程")
        df = _generate_synthetic_data(start_date, end_date)

    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 数据过滤
    if filter_min_pct > 0:
        df, _ = filter_stocks(df, min_pct=filter_min_pct)

    return df


def _generate_synthetic_data(start_date, end_date, n_stocks=50):
    """生成模拟的沪深300部分股票数据（用于测试完整流程）"""
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq="B")
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    industries = ["金融", "科技", "消费", "医药", "制造", "能源"]

    rows = []
    for code in codes:
        base_price = np.random.uniform(10, 100)
        base_pe = np.random.uniform(8, 60)
        base_vol = np.random.uniform(0.15, 0.45)
        momentum_factor = np.random.randn(len(dates)) * 0.02
        momentum_factor = np.cumsum(momentum_factor)
        industry = np.random.choice(industries)

        for i, date in enumerate(dates):
            noise = np.random.randn() * 0.015
            price = base_price * (1 + 0.05 * momentum_factor[i] + noise)
            price = max(price, 1.0)
            pe = base_pe + np.random.randn() * 5
            vol_60 = base_vol + np.random.randn() * 0.03
            vol_60 = max(0.05, vol_60)

            rows.append({
                "date": date,
                "code": code,
                "close": round(price, 2),
                "pe": round(pe, 2),
                "volume": int(np.random.uniform(1e6, 1e8)),
                "industry": industry,
            })

    df = pd.DataFrame(rows)
    return df


def calc_forward_return(df, periods=5):
    """计算未来N日收益"""
    df = df.copy()
    df = df.sort_values(["code", "date"])
    df[f"fwd_ret_{periods}d"] = (
        df.groupby("code")["close"]
        .transform(lambda x: x.shift(-periods) / x - 1)
    )
    return df
