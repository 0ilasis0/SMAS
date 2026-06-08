from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from base import MLTool
from debug import dbg
from ml.const import MLCol
from ml.data.market_features import MarketFeatureCol
from ml.params import MarketLGBMConfig, MarketThresholdConfig, TrainConfig


class MarketTrainer:
    """
    大盤防禦模型訓練器 (LightGBM)。
    預測未來幾天大盤是否會發生大跌，輸出 0~1 的危險機率，並轉換為「安全機率」供下游使用。
    """
    def __init__(self, config: MarketLGBMConfig = None):
        self.config = config or MarketLGBMConfig()
        # 用來紀錄 CV 過程中得到的最佳迭代次數
        self.optimal_trees = self.config.n_estimators

        # 平均 AUC，預設為0.5
        self.cv_avg_auc: float = 0.5

    def train_with_cv(self, df_clean: pd.DataFrame, lookahead: int, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        dbg.log(f"開始執行 LightGBM 大盤防禦模型 CV (Fold={n_splits}, Gap={lookahead})...")

        features = MarketFeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[MarketFeatureCol.TARGET_DANGER].astype(int)

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=lookahead)
        oof_predictions = pd.Series(index=X.index, dtype=float)

        cv_aucs = []
        best_iters = [] # 紀錄每個 Fold 的最佳停止點
        cv_importances = [] # 紀錄大盤特徵重要性

        lgbm_params = self.config.to_dict

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            # 將原本的 val_index 切成兩半，前半做驗證，後半做測試
            split_point = len(val_index) // 2

            # 在 Early Stop 和 Test 之間挖出 lookahead 的安全護城河
            early_stop_end = split_point - lookahead

            # 避免切分後樣本過少被掏空
            if early_stop_end <= 0 or split_point >= len(val_index):
                dbg.war(f"Fold {fold+1}: 樣本數不足以切割三階段，跳過此 Fold。")
                continue

            early_stop_index = val_index[:early_stop_end]
            test_index = val_index[split_point:]

            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[early_stop_index], y.iloc[early_stop_index]

            # 宣告 y_test 供後續驗證使用
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            scale_weight = MLTool.calculate_scale_weight(y_train)

            model = lgb.LGBMClassifier(**lgbm_params, scale_pos_weight=scale_weight)

            callbacks = [
                lgb.early_stopping(stopping_rounds=TrainConfig.ML_EARLY_STOP_ROUND, verbose=False)
            ]

            # 這裡只給它看 X_val，絕不給它看 X_test
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

            #取得被 scale_pos_weight 膨脹的危險機率
            y_pred_proba_danger_weighted = model.predict_proba(X_test)[:, 1]

            # 校正還原回同一基準的真實機率！
            y_pred_proba_danger = MLTool.unscale_probability(y_pred_proba_danger_weighted, scale_weight)

            oof_predictions.iloc[test_index] = y_pred_proba_danger

            # 紀錄最佳迭代次數
            if model.best_iteration_ is not None:
                best_iters.append(model.best_iteration_)

            if hasattr(model, 'feature_importances_'):
                cv_importances.append(model.feature_importances_)

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba_danger)
                cv_aucs.append(auc)
                dbg.log(f"Fold {fold+1}: 崩盤預測 AUC = {auc:.4f} (最佳樹量: {model.best_iteration_})")

        # 計算並保存平均最佳迭代次數，供 final_model 使用
        if best_iters:
            self.optimal_trees = int(np.mean(best_iters))
            dbg.log(f"💡 CV 判定最佳平均樹量為: {self.optimal_trees} 棵 (原設定 {self.config.n_estimators} 棵)")

        avg_auc = np.mean(cv_aucs) if cv_aucs else 0
        self.cv_avg_auc = avg_auc
        dbg.log(f"【Market Brain CV 結果】平均崩盤預測 AUC: {avg_auc:.4f}")

        if cv_importances:
            avg_importance = np.mean(cv_importances, axis=0)
            importance_series = pd.Series(avg_importance, index=features).sort_values(ascending=False)

            dbg.log("\n🏆 【Market Brain 崩盤預測核心特徵 (Top 5)】")
            for idx, (feat_name, imp_score) in enumerate(importance_series.head(5).items(), 1):
                dbg.log(f"  {idx}. {feat_name}: {imp_score:.4f}")

            dbg.log("\n🗑️ 【Market Brain 貢獻度最低特徵 (Bottom 5)】")
            for idx, (feat_name, imp_score) in enumerate(importance_series.tail(5).items(), 1):
                # 倒數排名，所以 idx 標示方式稍微不同
                dbg.log(f"  倒數 {6-idx}. {feat_name}: {imp_score:.4f}")

        return oof_predictions.dropna()

    def train_and_save_final_model(self, df_clean: pd.DataFrame, oof_preds: pd.Series, save_path: Path | str,
                                   thresh_config: MarketThresholdConfig = MarketThresholdConfig()):
        dbg.log(f"開始訓練最終上線版 LightGBM 大盤模型 (使用動態最佳樹量: {self.optimal_trees})...")
        features = MarketFeatureCol.get_features()
        X = df_clean[features]
        y = df_clean[MarketFeatureCol.TARGET_DANGER].astype(int)

        scale_weight = MLTool.calculate_scale_weight(y)

        lgbm_params = self.config.to_dict
        lgbm_params[MLCol.N_ESTIMATORS] = self.optimal_trees

        # ==========================================
        # 1. 訓練模型 (這時候讓它看全部資料)
        # ==========================================
        final_model = lgb.LGBMClassifier(**lgbm_params, scale_pos_weight=scale_weight)
        final_model.fit(X, y)

        # ==========================================
        # 2. 使用乾淨的 OOF 數據來評估 AUC 與門檻
        # ==========================================
        # 確保 OOF 的 Index 與原始資料的 y 對齊
        y_true_oof = y.loc[oof_preds.index].values
        y_prob_oof = oof_preds.values

        # 紀錄客觀無洩漏的 AUC
        final_model.val_auc = self.cv_avg_auc
        # 根據 F-beta 尋找最佳防禦門檻 (使用 OOF 計算！)
        final_model.dynamic_threshold = self._tune_danger_threshold(y_true_oof, y_prob_oof, thresh_config)

        final_model.train_scale_weight = float(scale_weight)

        dbg.log(f"✅ 已將真實 OOF 門檻 [{final_model.dynamic_threshold:.4f}] 寫入模型基因中 (AUC: {final_model.val_auc:.4f})。")

        # ==========================================
        # 3. 儲存具備最強權重與真實防禦參數的終極模型
        # ==========================================
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, str(save_path_obj))
        dbg.log(f"大盤防禦模型已成功儲存至: {save_path}")

    @staticmethod
    def _tune_danger_threshold(y_true: np.ndarray, y_prob: np.ndarray, config: MarketThresholdConfig) -> float:
        """大盤專用：根據 F-beta 尋找最佳防禦門檻"""
        # 防線 1：先檢查標籤是否單一，避免 sklearn 拋出錯誤
        if len(np.unique(y_true)) < 2:
            dbg.war("OOF 標籤單一，無法計算 PR 曲線，啟用退回門檻。")
            return config.FALLBACK_THRESHOLD

        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            beta = config.BETA

            numerator = (1 + beta**2) * (precisions * recalls)
            denominator = (beta**2 * precisions) + recalls + 1e-9
            f_scores = numerator / denominator

            # 防線 2：F-Score 全軍覆沒防護
            if np.max(f_scores) == 0:
                dbg.war("模型 PR 預測完全失效 (F-Score全為0)，啟用退回門檻。")
                return config.FALLBACK_THRESHOLD

            best_idx = np.argmax(f_scores[:-1])
            best_thresh = float(thresholds[best_idx])

            # 防線 3：極端高機率防護
            if best_thresh > 1.0:
                return config.FALLBACK_THRESHOLD

            # 防線 4：系統絕對安全底線
            final_thresh = max(best_thresh, config.ABS_MIN_THRESHOLD)

            if best_thresh < config.ABS_MIN_THRESHOLD:
                dbg.war(f"算出最佳門檻 {best_thresh:.4f} 過低，已強制拉升至底線 {config.ABS_MIN_THRESHOLD:.2f}")

            return final_thresh

        except Exception as e:
            dbg.war(f"動態門檻計算發生異常，使用預設值。詳細錯誤: {e}")
            return config.FALLBACK_THRESHOLD

    @staticmethod
    def load_inference_model(model_path: Path | str) -> lgb.LGBMClassifier:
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                dbg.error(f"大盤模型載入失敗: 找不到檔案 {model_path}")
                return None

            model = joblib.load(model_path)
            dbg.log("成功載入 LightGBM 大盤防禦模型。")
            return model
        except Exception as e:
            dbg.error
