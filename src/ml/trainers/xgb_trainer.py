from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from base import MathTool, MLTool
from debug import dbg
from ml.const import FeatureCol, MLCol
from ml.params import TrainConfig, XGBHyperParams


class XGBTrainer:
    """
    XGBoost 離線訓練器。
    包含滾動式時間窗交叉驗證 (TimeSeriesSplit) 與最終模型儲存。
    """
    def __init__(
            self,
            ticker: str,
            hp: XGBHyperParams = XGBHyperParams()
        ):
        self.ticker = ticker
        self.params = asdict(hp)
        self.optimal_trees = self.params.get(MLCol.N_ESTIMATORS, 100)

    def train_with_cv(self, df_clean: pd.DataFrame, lookahead: int, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        n_splits = MathTool.clamp(n_splits, TrainConfig.N_SPLITS_MIN, TrainConfig.N_SPLITS_MAX)
        dbg.log(f"開始執行 XGBoost TimeSeriesSplit 交叉驗證 (Fold={n_splits}, Gap={lookahead})...")

        features = FeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[FeatureCol.TARGET].astype(int)

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=lookahead)

        cv_accuracies, cv_aucs, cv_importances, best_iters = [], [], [], []
        oof_predictions = pd.Series(index=X.index, dtype=float)

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            split_point = len(val_index) // 2
            early_stop_end = split_point - lookahead

            if early_stop_end <= 0 or split_point >= len(val_index):
                dbg.war(f"Fold {fold+1}: 樣本不足以切割三階段，跳過。")
                continue

            early_stop_index = val_index[:early_stop_end]
            test_index = val_index[split_point:]

            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[early_stop_index], y.iloc[early_stop_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            scale_weight = MLTool.calculate_scale_weight(y_train)

            model = xgb.XGBClassifier(
                **self.params,
                scale_pos_weight=scale_weight,
                early_stopping_rounds=TrainConfig.EARLY_STOP_ROUND
            )

            # 僅使用 X_val 進行 Early Stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 僅預測完全沒看過的 X_test
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            oof_predictions.iloc[test_index] = y_pred_proba

            best_iters.append(model.best_iteration)

            acc = accuracy_score(y_test, y_pred)
            cv_accuracies.append(acc)

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
                cv_aucs.append(auc)
                dbg.log(f"Fold {fold+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f} (最佳樹量: {model.best_iteration})")

            cv_importances.append(model.feature_importances_)

        if best_iters:
            self.optimal_trees = int(np.mean(best_iters))
            dbg.log(f"💡 CV 判定最佳平均樹量為: {self.optimal_trees} 棵")

        avg_acc = np.mean(cv_accuracies) if cv_accuracies else 0
        avg_auc = np.mean(cv_aucs) if cv_aucs else 0
        dbg.log(f"【CV 驗證結果】平均 Accuracy: {avg_acc:.4f}, 平均 AUC: {avg_auc:.4f}")

        # if cv_importances:
        #     avg_importance = np.mean(cv_importances, axis=0)
            # importance_series = pd.Series(avg_importance, index=features).sort_values(ascending=False)

            # dbg.log("\n🏆 【XGBoost 核心預測特徵 (Top 10)】")
            # for idx, (feat_name, imp_score) in enumerate(importance_series.head(10).items(), 1):
            #     dbg.log(f"  {idx}. {feat_name}: {imp_score:.4f}")
            # dbg.log("-" * 40)

        return oof_predictions.dropna()

    def train_and_save_final_model(self, df_clean: pd.DataFrame, save_path: Path):
        dbg.log(f"開始訓練最終上線版 XGBoost 模型 (動態樹量={self.optimal_trees})...")

        features = FeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[FeatureCol.TARGET].astype(int)

        scale_weight = MLTool.calculate_scale_weight(y)

        # 套用 CV 計算出的最佳樹量
        final_params = self.params.copy()
        final_params[MLCol.N_ESTIMATORS] = self.optimal_trees

        final_model = xgb.XGBClassifier(**final_params, scale_pos_weight=scale_weight)
        final_model.fit(X, y)

        # 先轉成 Path 物件來建立資料夾
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(save_path_obj))

        dbg.log(f"最終模型已成功儲存至: {save_path}")

    @staticmethod
    def load_inference_model(model_path: Path | str) -> xgb.XGBClassifier:
        try:
            # 確保型態為 Path
            model_path = Path(model_path)

            if not model_path.exists():
                dbg.error(f"XGBoost模型載入失敗: 找不到檔案 {model_path}")
                return None

            model = xgb.XGBClassifier()
            model.load_model(model_path)
            dbg.log(f"成功載入 XGBoost 模型: {model_path}")
            return model

        except Exception as e:
            dbg.error(f"XGBoost 模型載入發生未知例外 [{type(e).__name__}]: {str(e)} \n目標路徑: {model_path}")
            return None
