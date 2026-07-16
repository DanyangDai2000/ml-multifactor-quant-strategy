"""
main.py - 沪深300多因子选股策略主入口

运行完整流程：
1. 加载数据
2. 因子计算 & IC 分析
3. K-means 聚类
4. Walk-Forward 模型验证 (LightGBM vs Ridge)
5. Backtrader 回测
6. 结果输出
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, calc_forward_return
from src.factors import calc_factors, process_factors, calc_ic_analysis
from src.clustering import cluster_stocks, get_best_cluster
from src.models import WalkForwardValidator, predict_scores
from src.backtest import run_backtest
from src.utils import calc_sharpe, calc_max_drawdown, calc_annual_return

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(name, content):
    """保存中间结果到 results/ 目录"""
    path = os.path.join(RESULTS_DIR, name)
    if isinstance(content, pd.DataFrame):
        content.to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            f.write(str(content))
    print(f"  [保存] {path}")


def main():
    print("=" * 60)
    print("  沪深300多因子选股策略 - 升级版")
    print("  LightGBM + Ridge + Walk-Forward + Backtrader")
    print("=" * 60)

    # ============================
    # 1. 数据加载
    # ============================
    print("\n[1/5] 加载数据...")
    df = load_data(filepath="data/hs300_2018_2022.csv")
    df = calc_forward_return(df, periods=5)
    print(f"  数据维度: {df.shape}")
    print(f"  时间范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  股票数量: {df['code'].nunique()}")

    # ============================
    # 2. 因子计算 & IC 分析
    # ============================
    print("\n[2/5] 因子计算 & IC 分析...")
    df = calc_factors(df)
    df = process_factors(df)

    ic_df, ic_summary, ic_yearly = calc_ic_analysis(df)

    print("\n  IC 汇总统计:")
    print(ic_summary.to_string(index=False))

    print("\n  分年度 IC:")
    print(ic_yearly.to_string(index=False))

    save_results("ic_daily.csv", ic_df)
    save_results("ic_summary.csv", ic_summary)
    save_results("ic_yearly.csv", ic_yearly)

    # ============================
    # 3. K-means 聚类
    # ============================
    print("\n[3/5] K-means 聚类...")
    df = cluster_stocks(df, n_clusters=4)
    n_clustered = df["cluster"].notna().sum()
    print(f"  已聚类样本数: {n_clustered}")

    # 找出最优聚类组
    best_cluster, cluster_perf = get_best_cluster(df)
    print(f"  最优聚类组: {best_cluster} (平均收益最高)")

    # ============================
    # 4. Walk-Forward 模型验证
    # ============================
    print("\n[4/5] Walk-Forward 模型验证...")

    feature_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]
    target_col = "fwd_ret_5d"

    # 添加聚类特征到特征列表
    if "cluster" in df.columns and df["cluster"].notna().any():
        df["cluster_feat"] = df["cluster"].fillna(-1).astype(float)
        feature_cols_with_cluster = feature_cols + ["cluster_feat"]
    else:
        feature_cols_with_cluster = feature_cols

    validator = WalkForwardValidator(
        train_window=504,  # ~2年交易日
        test_window=63,    # ~3个月
        mode="sliding"
    )

    results = {}

    for model_type in ["lgb", "ridge"]:
        print(f"\n  --- {model_type.upper()} ---")
        try:
            res = validator.validate(
                df,
                feature_cols=feature_cols_with_cluster,
                target_col=target_col,
                model_type=model_type,
            )
            results[model_type] = res
            print(f"  总 IC: {res['overall_ic']:.4f}")
            print(f"  窗口数: {len(res['window_metrics'])}")

            if res["feature_importances"] is not None:
                print("  特征重要性:")
                for name, imp in zip(res["feature_cols"], res["feature_importances"]):
                    print(f"    {name}: {imp:.4f}")

            # 保存特征重要性
            if res["feature_importances"] is not None:
                fi_df = pd.DataFrame({
                    "feature": res["feature_cols"],
                    "importance": res["feature_importances"],
                }).sort_values("importance", ascending=False)
                save_results(f"feature_importance_{model_type}.csv", fi_df)

            # 保存窗口指标
            if not res["window_metrics"].empty:
                save_results(f"window_metrics_{model_type}.csv", res["window_metrics"])

        except Exception as e:
            print(f"  [ERROR] {model_type} 失败: {e}")

    # ============================
    # 5. 回测
    # ============================
    print("\n[5/5] Backtrader 回测...")

    backtest_results = {}

    for model_type in ["lgb", "ridge"]:
        print(f"\n  --- {model_type.upper()} 回测 ---")
        try:
            # 生成预测得分
            df_scores = predict_scores(
                df,
                feature_cols=feature_cols_with_cluster,
                target_col=target_col,
                model_type=model_type,
                train_window=504,
            )
            print(f"  预测得分生成完成，样本数: {len(df_scores)}")

            # 运行回测
            _, summary = run_backtest(
                df_scores, top_pct=0.2, commission=0.001, cash=1_000_000
            )

            backtest_results[model_type] = summary

            print(f"  初始资金: {summary['initial_cash']:,.0f}")
            print(f"  最终资金: {summary['final_value']:,.0f}")
            print(f"  总收益: {summary['total_return']:.2f}%")
            if summary.get('annual_return') is not None:
                print(f"  年化收益: {summary['annual_return']:.2f}%")
            else:
                print(f"  年化收益: N/A")
            if summary.get('sharpe') is not None:
                print(f"  夏普比率: {summary['sharpe']:.4f}")
            else:
                print(f"  夏普比率: N/A")
            print(f"  最大回撤: {summary['max_drawdown']:.2f}%")

            # 保存净值序列
            if not summary["nav"].empty:
                nav_df = summary["nav"].reset_index()
                nav_df.columns = ["date", "nav"]
                save_results(f"nav_{model_type}.csv", nav_df)

        except Exception as e:
            print(f"  [ERROR] {model_type} 回测失败: {e}")
            import traceback
            traceback.print_exc()

    # ============================
    # 结果对比汇总
    # ============================
    print("\n" + "=" * 60)
    print("  策略对比汇总")
    print("=" * 60)

    compare_rows = []
    for mt in ["lgb", "ridge"]:
        if mt in results and results[mt] is not None:
            row = {
                "模型": mt.upper(),
                "预测IC": f"{results[mt]['overall_ic']:.4f}",
            }
            if mt in backtest_results:
                s = backtest_results[mt]
                row["年化收益"] = f"{s['annual_return']:.2f}%" if s['annual_return'] else "N/A"
                row["夏普"] = f"{s['sharpe']:.2f}" if s['sharpe'] else "N/A"
                row["最大回撤"] = f"{s['max_drawdown']:.2f}%"
            compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    print("\n", compare_df.to_string(index=False))

        # ============================
    # 6. 可视化图表
    # ============================
    from src.plot_results import plot_all

    # 组装净值字典
    nav_dict = {}
    for mt in ["lgb", "ridge"]:
        s = backtest_results.get(mt)
        if s and s["nav"] is not None and len(s["nav"]) > 0:
            nav_dict[mt] = s["nav"]

    # 读取 IC 数据
    ic_path = os.path.join(RESULTS_DIR, "ic_daily.csv")
    ic_df = None
    if os.path.exists(ic_path):
        ic_df = pd.read_csv(ic_path, parse_dates=["date"])

    plot_all(ic_df, nav_dict, output_dir=RESULTS_DIR)

    print("\n✅ 全部流程完成！")
    print(f"   结果保存在: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
