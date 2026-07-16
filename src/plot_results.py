"""
plot_results.py - 回测结果可视化
生成三张图：
1. 收益曲线（策略 vs 基准）
2. IC 时间序列 + 移动平均
3. 分年度 IC 柱状图

用法：
    from src.plot_results import plot_all
    fig = plot_all(ic_df, nav_dict, output_dir="results")
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "PingFang SC",
                                     "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_nav_curve(nav_dict, output_dir="results"):
    """收益曲线（策略 vs 基准）"""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"lgb": "#2196F3", "ridge": "#FF9800", "hs300": "#4CAF50"}

    for model_type, nav in nav_dict.items():
        if nav is None or len(nav) == 0:
            continue
        nav = nav / nav.iloc[0] * 100  # 归一化到 100
        label = model_type.upper()
        if model_type == "hs300":
            label = "沪深300"
        ax.plot(nav.index, nav.values, label=label,
                color=colors.get(model_type, "#999"), linewidth=2 if model_type == "lgb" else 1.5)

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("策略净值曲线（归一化至100）", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("净值", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "chart_nav.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表] {path}")
    return fig


def plot_ic_timeseries(ic_df, output_dir="results"):
    """IC 时间序列图（含30日移动平均）"""
    if ic_df is None or len(ic_df) == 0:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    factors = ic_df["factor"].unique()
    colors = {"factor_pe": "#E91E63", "factor_momentum_20": "#2196F3",
              "factor_vol_60": "#4CAF50"}
    labels = {"factor_pe": "PE", "factor_momentum_20": "20日动量",
              "factor_vol_60": "60日波动率"}

    ic_df = ic_df.copy()
    ic_df["date"] = pd.to_datetime(ic_df["date"])
    ic_df = ic_df.sort_values("date")

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        sub = ic_df[ic_df["factor"] == factor]

        ax.plot(sub["date"], sub["IC"], alpha=0.3, color=colors.get(factor, "#999"),
                linewidth=0.8, label="日度IC")

        # 30日移动平均
        sub = sub.sort_values("date")
        sub["MA30"] = sub["IC"].rolling(30, min_periods=10).mean()
        ax.plot(sub["date"], sub["MA30"], color=colors.get(factor, "#999"),
                linewidth=2, label="30日MA")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{labels.get(factor, factor)} IC 时间序列", fontsize=12)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("日期", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, "chart_ic_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表] {path}")
    return fig


def plot_ic_yearly(ic_df, output_dir="results"):
    """分年度 IC 柱状图"""
    if ic_df is None or len(ic_df) == 0:
        return None

    ic_df = ic_df.copy()
    ic_df["year"] = pd.to_datetime(ic_df["date"]).dt.year

    yearly = ic_df.groupby(["factor", "year"])["IC"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    factors = yearly["factor"].unique()
    n_factors = len(factors)
    years = sorted(yearly["year"].unique())
    n_years = len(years)

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    width = 0.25

    for i, factor in enumerate(factors):
        sub = yearly[yearly["factor"] == factor]
        sub = sub.set_index("year").reindex(years).fillna(0)
        x = np.arange(n_years) + i * width
        bars = ax.bar(x, sub["IC"], width, label=factor,
                       color=colors[i % len(colors)], alpha=0.85)
        # 柱上标数值
        for bar, val in zip(bars, sub["IC"]):
            if not np.isnan(val) and abs(val) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.06,
                        f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=8, rotation=45)

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xticks(np.arange(n_years) + width * (n_factors - 1) / 2)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_title("分年度 IC 均值", fontsize=14, fontweight="bold")
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("IC", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, "chart_ic_yearly.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图表] {path}")
    return fig


def plot_all(ic_df, nav_dict, output_dir="results"):
    """生成全部图表"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n[绘图] 生成可视化图表...")

    fig1 = plot_nav_curve(nav_dict, output_dir)
    fig2 = plot_ic_timeseries(ic_df, output_dir)
    fig3 = plot_ic_yearly(ic_df, output_dir)

    print("  [绘图] 完成！")
    return [fig1, fig2, fig3]


if __name__ == "__main__":
    # 测试：从结果文件加载并绘图
    ic_path = "results/ic_daily.csv"
    if os.path.exists(ic_path):
        ic_df = pd.read_csv(ic_path, parse_dates=["date"])
        nav_dict = {}
        for nav_file in ["nav_lgb.csv", "nav_ridge.csv"]:
            p = f"results/{nav_file}"
            if os.path.exists(p):
                nav = pd.read_csv(p, parse_dates=["date"], index_col="date")["nav"]
                nav_dict[nav_file.replace("nav_", "").replace(".csv", "")] = nav
        plot_all(ic_df, nav_dict)
    else:
        print("结果文件不存在，请先运行 main.py")
