"""
backtest.py - 纯 pandas 向量化回测（替代 Backtrader）

策略逻辑：
- 每月第一个交易日调仓
- 根据预测得分排序，选 Top 20% 买入
- 等权重配置
- 双边 0.1% 手续费

优势：无外部依赖，更快，无 Backtrader 兼容性问题
"""

import numpy as np
import pandas as pd
import math


def run_backtest(df_with_scores, top_pct=0.2, commission=0.001, cash=1_000_000):
    """
    向量化回测

    参数:
    - df_with_scores: DataFrame，必须包含列 ['date', 'code', 'close', 'pred_score']
    - top_pct: 每月选股比例（选得分最高的 pct）
    - commission: 双边手续费率
    - cash: 初始资金

    返回:
    - cerebro: None (兼容 Backtrader 接口)
    - summary: dict 包含回测结果
    """
    df = df_with_scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)

    # 标记每月第一个交易日
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["year_month"] = df["year"] * 100 + df["month"]

    # 每月第一个交易日：取每组最小日期
    first_trade = df.groupby(["year_month", "code"])["date"].min().reset_index()
    first_trade = first_trade.groupby("year_month")["date"].min().reset_index()
    first_trade["is_rebalance"] = True
    df = df.merge(first_trade, on=["year_month", "date"], how="left")
    df["is_rebalance"] = df["is_rebalance"].fillna(False)

    # 只在调仓日操作
    rebalance_dates = sorted(df[df["is_rebalance"]]["date"].unique())
    print(f"  [回测] 调仓日: {len(rebalance_dates)} 天")

    # 逐月模拟
    portfolio = {"cash": cash, "positions": {}, "nav_history": []}

    for i, rdate in enumerate(rebalance_dates):
        # 当前调仓日的截面数据
        today_slice = df[(df["date"] == rdate) & (df["pred_score"].notna())].copy()
        if len(today_slice) == 0:
            continue

        # 按得分排序选股
        today_slice = today_slice.sort_values("pred_score", ascending=False)
        n_select = max(1, int(len(today_slice) * top_pct))
        selected = today_slice.head(n_select)

        # 先平旧仓（用上一期价格）
        close_value = 0
        for code, shares in list(portfolio["positions"].items()):
            if code in today_slice["code"].values:
                exit_price = today_slice[today_slice["code"] == code]["close"].iloc[0]
            else:
                # 找不到当天价格，找最近的
                prev = df[(df["code"] == code) & (df["date"] <= rdate)].sort_values("date")
                if len(prev) == 0:
                    continue
                exit_price = prev["close"].iloc[-1]
            close_value += shares * exit_price * (1 - commission)
            del portfolio["positions"][code]

        portfolio["cash"] += close_value

        # 建新仓
        invest_per_stock = portfolio["cash"] / n_select
        new_positions = {}
        for _, row in selected.iterrows():
            price = row["close"]
            if price <= 0 or math.isnan(price):
                continue
            shares = int(invest_per_stock / price)
            cost = shares * price * (1 + commission)
            if cost > 0:
                new_positions[row["code"]] = int(shares)
                portfolio["cash"] -= cost

        portfolio["positions"].update(new_positions)

        # 记录净值
        total_value = portfolio["cash"]
        for code, shares in portfolio["positions"].items():
            # 使用调仓日收盘价估值
            price_row = today_slice[today_slice["code"] == code]
            if len(price_row) > 0:
                total_value += shares * price_row["close"].iloc[0]
            else:
                prev = df[(df["code"] == code) & (df["date"] <= rdate)].sort_values("date")
                if len(prev) > 0:
                    total_value += shares * prev["close"].iloc[-1]

        portfolio["nav_history"].append({"date": rdate, "nav": total_value})

        if i < 3 or i == len(rebalance_dates) - 1:
            print(f"    [{i+1}/{len(rebalance_dates)}] {rdate.date()}: "
                  f"选{n_select}只, 净值={total_value:.0f}")

    # 生成净值序列
    nav_df = pd.DataFrame(portfolio["nav_history"])
    if not nav_df.empty:
        nav_df = nav_df.set_index("date")
        nav_df["nav"] = nav_df["nav"].astype(float)

    final_value = nav_df["nav"].iloc[-1] if not nav_df.empty else cash
    total_return = (final_value / cash - 1) * 100

    # 计算指标
    if len(nav_df) > 1:
        daily_returns = nav_df["nav"].pct_change().dropna()
        sharpe = np.sqrt(12) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        cum_max = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cum_max) / cum_max
        max_dd = drawdown.min() * 100

        years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
        annual_ret = ((final_value / cash) ** (1 / years) - 1) * 100 if years > 0 else 0
    else:
        sharpe = 0
        max_dd = 0
        annual_ret = 0

    print(f"  [回测结果]")
    print(f"    初始资金: {cash:,.0f}")
    print(f"    最终资金: {final_value:,.0f}")
    print(f"    总收益: {total_return:.2f}%")
    print(f"    年化收益: {annual_ret:.2f}%")
    print(f"    夏普比率: {sharpe:.4f}")
    print(f"    最大回撤: {max_dd:.2f}%")

    summary = {
        "initial_cash": cash,
        "final_value": final_value,
        "total_return": total_return,
        "sharpe": sharpe if not math.isnan(sharpe) else None,
        "max_drawdown": max_dd,
        "max_drawdown_len": 0,
        "annual_return": annual_ret,
        "nav": nav_df["nav"] if not nav_df.empty else pd.Series(dtype=float),
    }

    return None, summary
