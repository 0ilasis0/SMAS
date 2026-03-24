from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from base import MathTool
from debug import dbg
from ml.params import FeatureCol, TrainConfig, XGBHyperParams
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from path import PathConfig


class XGBTrainer:
    """
    XGBoost 離線訓練器。
    包含滾動式時間窗交叉驗證 (TimeSeriesSplit) 與最終模型儲存。
    """
    def __init__(
            self,
            model_save_path: str = PathConfig.XGB_MODEL,
            hp: XGBHyperParams = XGBHyperParams()
        ):
        self.model_save_path = Path(model_save_path)
        self.params = asdict(hp)

    def train_with_cv(self, df_clean: pd.DataFrame, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        """
        執行 TimeSeriesSplit 交叉驗證，評估模型在不同歷史階段的表現。
        """
        n_splits = MathTool.clamp(n_splits, TrainConfig.N_SPLITS_MIN, TrainConfig.N_SPLITS_MAX)

        dbg.log(f"開始執行 XGBoost TimeSeriesSplit 交叉驗證 (Fold={n_splits})...")

        # 切分特徵 (X) 與標籤 (y)
        features = FeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[FeatureCol.TARGET].astype(int)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_accuracies = []
        cv_aucs = []
        cv_importances = []

        # 存放 OOF 預測結果 (索引與原始資料對齊)
        oof_predictions = pd.Series(index=X.index, dtype=float)

        # 滾動訓練與驗證
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # 初始化並訓練模型
            model = xgb.XGBClassifier(
                **self.params,
                early_stopping_rounds=TrainConfig.EARLY_STOP_ROUND
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 預測驗證集
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # 收集 OOF 預測值
            oof_predictions.iloc[val_index] = y_pred_proba

            acc = accuracy_score(y_val, y_pred)
            cv_accuracies.append(acc)

            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, y_pred_proba)
                cv_aucs.append(auc)
                auc_str = f"{auc:.4f}"
            else:
                auc_str = "N/A (單一類別)"

            cv_importances.append(model.feature_importances_)
            dbg.log(f"Fold {fold+1}: Accuracy = {acc:.4f}, AUC = {auc_str}")

        if not cv_accuracies:
            dbg.error("交叉驗證失敗：資料量可能過少，無法進行 TimeSeriesSplit。")
            return pd.Series(dtype=float)

        # 搭配特徵名稱印出排行榜，計算平均重要性
        feature_importances = np.mean(cv_importances, axis=0)
        avg_acc = np.mean(cv_accuracies)
        avg_auc = np.mean(cv_aucs)

        importance_dict = dict(zip(features, feature_importances))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        dbg.log("【特徵重要性排行榜前五名】")
        for feat, imp in sorted_importance[:5]:
            dbg.log(f"- {feat}: {imp:.4f}")

        dbg.log(f"【CV 驗證結果】平均 Accuracy: {avg_acc:.4f}, 平均 AUC: {avg_auc:.4f}")

        # 回傳移除 NaN 的 OOF 預測結果 (最前面的資料因為沒被當作驗證集過，所以會是 NaN)
        return oof_predictions.dropna()

    def train_and_save_final_model(self, df_clean: pd.DataFrame):
        """
        使用【所有】的歷史資料訓練最終上線版本的模型，並匯出權重檔。
        """
        dbg.log("開始訓練最終上線版 XGBoost 模型 (使用全量資料)...")

        features = FeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[FeatureCol.TARGET].astype(int)

        # 訓練全量模型
        final_model = xgb.XGBClassifier(**self.params)
        final_model.fit(X, y)

        # 確保儲存目錄存在
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # 匯出權重
        final_model.save_model(self.model_save_path)
        dbg.log(f"最終模型已成功儲存至: {self.model_save_path}")

    @staticmethod
    def load_inference_model(model_path: str) -> xgb.XGBClassifier:
        """
        【供 UI 推論端使用】載入訓練好的模型權重。
        """
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            dbg.log(f"成功載入 XGBoost 模型: {model_path}")
            return model
        except Exception as e:
            dbg.error(f"模型載入失敗: {e}")
            return None
