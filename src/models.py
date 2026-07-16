"""
models.py - LightGBM vs Ridge Walk-Forward 模型训练与评估
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# LightGBM 可选（需要 libomp 编译支持）
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
except OSError:
    _HAS_LGB = False

if not _HAS_LGB:
    import warnings
    warnings.warn("LightGBM 不可用，使用 RandomForest 替代（完整功能需 brew install libomp）")


class WalkForwardValidator:
    """
    Walk-Forward 滚动验证
    支持 expanding window 和 sliding window

    用法:
        validator = WalkForwardValidator(
            train_window=252,    # 训练窗口大小（交易日）
            test_window=63,      # 验证窗口大小（约3个月）
            mode="sliding"       # 'sliding' 或 'expanding'
        )
        lgb_results = validator.validate(df, model_type="lgb")
        ridge_results = validator.validate(df, model_type="ridge")
    """

    def __init__(self, train_window=504, test_window=63, mode="sliding"):
        """
        Parameters:
        - train_window: 训练集窗口（交易日数）
        - test_window: 验证集窗口
        - mode: "sliding" | "expanding"
          sliding = 固定窗口滑动
          expanding = 训练集不断累积
        """
        self.train_window = train_window
        self.test_window = test_window
        self.mode = mode

    def get_windows(self, dates):
        """生成 train/test 索引对"""
        dates = sorted(dates)
        windows = []
        start = 0
        while start + self.train_window + self.test_window <= len(dates):
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            windows.append({
                "train": dates[start:train_end],
                "test": dates[train_end:test_end],
            })
            if self.mode == "sliding":
                start += self.test_window
            else:
                # expanding: 训练集左端不动，右侧扩展
                start = 0
                self.train_window += self.test_window
        return windows

    def prepare_data(self, df, feature_cols, target_col, train_dates, test_dates):
        """切分训练/测试数据"""
        train = df[df["date"].isin(train_dates)].dropna(
            subset=feature_cols + [target_col]
        )
        test = df[df["date"].isin(test_dates)].dropna(
            subset=feature_cols + [target_col]
        )

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        return X_train, y_train, X_test, y_test, train, test

    def validate(self, df, feature_cols=None, target_col="fwd_ret_5d",
                 model_type="lgb", lgb_params=None, ridge_alpha=1.0):
        """
        执行 Walk-Forward 验证

        参数:
        - model_type: "lgb" | "ridge"
        - lgb_params: LightGBM 超参数
        - ridge_alpha: Ridge 正则化强度

        返回:
        - results: dict 包含每期预测、真实值、特征重要性、评价指标等
        """
        if feature_cols is None:
            feature_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]

        dates = sorted(df["date"].unique())
        windows = self.get_windows(dates)

        if not windows:
            raise ValueError("数据长度不足以进行 Walk-Forward 验证")

        all_y_true = []
        all_y_pred = []
        all_test_dates = []
        all_test_codes = []
        feature_importances = []
        window_metrics = []

        for i, w in enumerate(windows):
            X_train, y_train, X_test, y_test, train_df, test_df = self.prepare_data(
                df, feature_cols, target_col, w["train"], w["test"]
            )

            if len(X_train) < 100 or len(X_test) < 10:
                continue

            if model_type == "lgb":
                if _HAS_LGB:
                    params = lgb_params or {
                        "n_estimators": 200,
                        "learning_rate": 0.05,
                        "max_depth": 4,
                        "num_leaves": 16,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": 42,
                        "verbose": -1,
                    }
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train)
                else:
                    # 无 LightGBM 时用 RandomForest 替代
                    model = RandomForestRegressor(
                        n_estimators=200, max_depth=6,
                        random_state=42, n_jobs=-1
                    )
                    model.fit(X_train, y_train)

                if hasattr(model, "feature_importances_"):
                    feat_imp = model.feature_importances_
                else:
                    feat_imp = np.zeros(len(feature_cols))

            else:  # ridge
                model = Ridge(alpha=ridge_alpha, random_state=42)
                model.fit(X_train, y_train)
                # Ridge 用系数绝对值作为"重要性"
                feat_imp = np.abs(model.coef_)

            y_pred = model.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_test_dates.extend(test_df["date"].tolist())
            all_test_codes.extend(test_df["code"].tolist())
            feature_importances.append(feat_imp)

            # 当期指标
            ic = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
            mse = mean_squared_error(y_test, y_pred)
            window_metrics.append({
                "window": i,
                "train_start": w["train"][0],
                "train_end": w["train"][-1],
                "test_start": w["test"][0],
                "test_end": w["test"][-1],
                "test_ic": ic,
                "test_mse": mse,
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

        # 汇总
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        overall_ic = np.corrcoef(all_y_true, all_y_pred)[0, 1] if len(all_y_true) > 1 else 0

        results = {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "dates": all_test_dates,
            "codes": all_test_codes,
            "overall_ic": overall_ic,
            "feature_importances": np.mean(feature_importances, axis=0) if feature_importances else None,
            "feature_cols": feature_cols,
            "window_metrics": pd.DataFrame(window_metrics) if window_metrics else pd.DataFrame(),
            "model_type": model_type,
        }

        return results


def predict_scores(df, feature_cols=None, target_col="fwd_ret_5d",
                   model_type="lgb", train_window=504):
    """
    简便方法：用最后 train_window 个交易日训练，对全部数据打分
    用于回测生成信号
    """
    if feature_cols is None:
        feature_cols = ["factor_pe", "factor_momentum_20", "factor_vol_60"]

    df = df.copy().sort_values("date")
    unique_dates = sorted(df["date"].unique())

    # 用最早的数据训练一个全局模型（仅在演示中）
    # 实际回测应滚动训练
    train_dates = unique_dates[:train_window] if len(unique_dates) > train_window else unique_dates[:-63]
    if len(train_dates) < 100:
        train_dates = unique_dates[:max(100, len(unique_dates) // 2)]

    train = df[df["date"].isin(train_dates)].dropna(subset=feature_cols + [target_col])
    X_train = train[feature_cols].values
    y_train = train[target_col].values

    if model_type == "lgb":
        if _HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, num_leaves=16,
                random_state=42, verbose=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200, max_depth=6,
                random_state=42, n_jobs=-1
            )
    else:
        model = Ridge(alpha=1.0, random_state=42)

    model.fit(X_train, y_train)

    # 对所有数据预测
    all_data = df.dropna(subset=feature_cols + [target_col]).copy()
    X_all = all_data[feature_cols].values
    all_data["pred_score"] = model.predict(X_all)

    return all_data
