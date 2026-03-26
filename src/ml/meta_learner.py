import math

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from base import MathTool
from debug import dbg
from ml.params import MetaCol, MetaHyperParams, TrainConfig
from path import PathConfig


class MetaLearner:
    """
    Level 1 Meta-Learner (元學習器)。
    負責整合 Level 0 (XGBoost & DL) 的 OOF 預測機率，輸出最終綜合勝率。
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model_save_path = PathConfig.get_meta_model_path(self.ticker)

        self.model = LogisticRegression(
            C=MetaHyperParams.C,
            class_weight=MetaHyperParams.CLASS_WEIGHT,
            penalty=MetaHyperParams.PENALTY,
            solver=MetaHyperParams.SOLVER,
            random_state=MetaHyperParams.RANDOM_STATE,
            max_iter=MetaHyperParams.MAX_ITER
        )

    def train_and_evaluate(self, oof_xgb: pd.Series, oof_dl: pd.Series, y_true: pd.Series):
        """
        訓練 Meta-Learner 並評估整合後的表現。
        :param oof_xgb: XGBoost 產出的 OOF 機率 (附帶 DatetimeIndex)
        :param oof_dl: DL 產出的 OOF 機率 (附帶 DatetimeIndex)
        :param y_true: 真實的標籤 Target (附帶 DatetimeIndex)
        """
        dbg.log("開始進行 Meta-Learner (Level 1) 整合訓練...")

        # 對齊資料
        df_meta = pd.concat([oof_xgb, oof_dl, y_true], axis=1, join='inner')
        df_meta.columns = [MetaCol.PROB_XGB, MetaCol.PROB_DL, MetaCol.TARGET]

        if df_meta.empty:
            dbg.error("資料對齊後為空，請檢查 OOF 預測值的 Index 是否正確！")
            return

        dbg.log(f"成功對齊 OOF 資料，共獲得 {len(df_meta)} 筆有效整合樣本。")

        X_meta = df_meta[[MetaCol.PROB_XGB, MetaCol.PROB_DL]]
        y_meta = df_meta[MetaCol.TARGET].astype(int)

        tscv = TimeSeriesSplit(n_splits=TrainConfig.N_SPLITS)
        honest_aucs = cross_val_score(
            self.model,
            X_meta,
            y_meta,
            cv=tscv,
            scoring='roc_auc'
        )
        dbg.log(f"【Meta-Learner 誠實評估】5-Fold AUC: {honest_aucs.mean():.4f} (+/- {honest_aucs.std()*2:.4f})")

        # 訓練邏輯迴歸
        self.model.fit(X_meta, y_meta)

        # 觀察大腦權重分配
        weights = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        dbg.log("【決策權重分配】")
        dbg.log(f" ➔ XGBoost 權重: {weights[0]:.4f}")
        dbg.log(f" ➔ DL 模型 權重: {weights[1]:.4f}")
        dbg.log(f" ➔ 基礎截距 (偏誤): {intercept:.4f}")

        # 儲存模型
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_save_path)
        dbg.log(f"Meta-Learner 權重已儲存至: {self.model_save_path}")

    def load_inference_model(self):
        """【供 UI 推論端使用】載入訓練好的 Meta-Learner"""
        try:
            self.model = joblib.load(self.model_save_path)
            dbg.log(f"成功載入 Meta-Learner: {self.model_save_path}")
        except Exception as e:
            dbg.error(f"Meta-Learner 載入失敗: {e}")

    # def predict_final_probability(self, prob_xgb: float, prob_dl: float) -> float:
    #     """
    #     給線上實戰 (UI / 階段三行為樹) 呼叫用。
    #     輸入雙腦機率，輸出最終決定性的勝率。
    #     """
    #     if math.isnan(prob_xgb) or math.isnan(prob_dl):
    #         dbg.war("接收到 NaN 機率，回傳中性勝率 0.5")
    #         return 0.5

    #     prob_xgb = MathTool.clamp(prob_xgb, 0.0, 1.0)
    #     prob_dl = MathTool.clamp(prob_dl, 0.0, 1.0)

    #     X_new = pd.DataFrame({MetaCol.PROB_XGB: [prob_xgb], MetaCol.PROB_DL: [prob_dl]})
    #     return self.model.predict_proba(X_new)[0, 1]

    def predict_final_probability(self, prob_xgb: float, prob_dl: float) -> float:
        """
        給線上實戰 (UI / 階段三行為樹) 呼叫用。
        輸入雙腦機率，輸出最終決定性的勝率。
        """
        if math.isnan(prob_xgb) or math.isnan(prob_dl):
            dbg.war("接收到 NaN 機率，回傳中性勝率 0.5")
            return 0.5

        prob_xgb = MathTool.clamp(prob_xgb, 0.0, 1.0)
        prob_dl = MathTool.clamp(prob_dl, 0.0, 1.0)
        return (prob_xgb + prob_dl) / 2.0
