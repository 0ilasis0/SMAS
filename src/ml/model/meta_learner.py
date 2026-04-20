import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from base import MathTool
from debug import dbg
from ml.const import SignalCol
from ml.params import MetaHyperParams, TrainConfig


class MetaLearner:
    """
    Level 1 Meta-Learner (元學習器)。
    負責整合 Level 0 (XGBoost & DL) 的 OOF 預測機率，輸出最終綜合勝率。
    """
    def __init__(self, ticker: str, hp: MetaHyperParams = MetaHyperParams()):
        self.ticker = ticker

        self.hp = hp
        self.model = LogisticRegression(
            C=self.hp.C,
            class_weight=self.hp.CLASS_WEIGHT,
            penalty=self.hp.PENALTY,
            solver=self.hp.SOLVER,
            random_state=self.hp.RANDOM_STATE,
            max_iter=self.hp.MAX_ITER
        )

    def evaluate_oof(self, aligned_oof_xgb: pd.Series, aligned_oof_dl: pd.Series, aligned_y_true: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        接收已對齊雙腦的 OOF 預測，並執行交叉驗證評估。
        :return: (X_meta, y_meta) 供後續最終訓練使用
        """
        dbg.log("開始進行 Meta-Learner (Level 1) OOF 整合評估...")

        X_meta = pd.DataFrame({
            SignalCol.PROB_XGB: aligned_oof_xgb,
            SignalCol.PROB_DL: aligned_oof_dl
        })
        y_meta = aligned_y_true.astype(int)

        if X_meta.empty:
            dbg.error("接收到的 OOF 資料為空！")
            return pd.DataFrame(), pd.Series()

        dbg.log(f"成功接收 OOF 資料，共獲得 {len(X_meta)} 筆有效整合樣本。")

        # 誠實評估 (CV)
        tscv = TimeSeriesSplit(n_splits=TrainConfig.N_SPLITS)
        try:
            honest_aucs = cross_val_score(
                self.model,
                X_meta,
                y_meta,
                cv=tscv,
                scoring='roc_auc'
            )
            dbg.log(f"【Meta-Learner 誠實評估】5-Fold AUC: {honest_aucs.mean():.4f} (+/- {honest_aucs.std()*2:.4f})")
        except ValueError as e:
            dbg.war(f"【Meta-Learner 誠實評估】CV 驗證失敗 (可能是樣本標籤過度單一): {e}")

        return X_meta, y_meta

    def train_and_save_final_model(self, X_meta: pd.DataFrame, y_meta: pd.Series, save_path: Path | str):
        """
        使用對齊後的完整 OOF 資料訓練最終邏輯迴歸權重，並存檔。
        """
        if X_meta.empty or y_meta.empty:
            dbg.error("無效的 Meta-Learner 訓練資料，取消存檔。")
            return

        dbg.log("開始訓練最終上線版 Meta-Learner 模型...")

        # 訓練邏輯迴歸
        self.model.fit(X_meta, y_meta)

        # 觀察大腦權重分配
        weights = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        dbg.log("【決策權重分配】")
        dbg.log(f" ➔ XGBoost 權重: {weights[0]:.4f}")
        dbg.log(f" ➔ DL 模型 權重: {weights[1]:.4f}")
        dbg.log(f" ➔ 基礎截距 (偏誤): {intercept:.4f}")

        # 使用外部傳入的路徑儲存
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 寫入檔案時，強制轉回字串交給 joblib
        joblib.dump(self.model, str(save_path_obj))
        dbg.log(f"Meta-Learner 權重已儲存至: {save_path}")

    def load_inference_model(self, model_path: Path | str) -> bool:
        """【供 UI 推論端使用】載入訓練好的 Meta-Learner"""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                dbg.error(f"Meta-Learner 載入失敗: 找不到檔案 {model_path}")
                return False

            self.model = joblib.load(model_path)
            dbg.log(f"成功載入 Meta-Learner: {model_path}")
            return True
        except Exception as e:
            dbg.error(f"Meta-Learner 載入發生未知例外 [{type(e).__name__}]: {e}")
            return False

    def predict_final_probability(self, prob_xgb: float, prob_dl: float) -> float:
        """
        根據 XGBoost 與 DL 的預測機率，透過 Logistic Regression 產出最終綜合機率。
        """
        if math.isnan(prob_xgb) or math.isnan(prob_dl):
            dbg.war("接收到 NaN 機率，回傳中性勝率 0.5")
            return 0.5

        prob_xgb = MathTool.clamp(prob_xgb, 0.0, 1.0)
        prob_dl = MathTool.clamp(prob_dl, 0.0, 1.0)

        X_new = pd.DataFrame(
            [[prob_xgb, prob_dl]],
            columns=[SignalCol.PROB_XGB, SignalCol.PROB_DL]
        )
        return self.model.predict_proba(X_new)[0, 1]