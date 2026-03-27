import gc

import joblib
import torch

from data.const import StockCol, TimeUnit
from data.fetcher import Fetcher
from data.manager import DataManager
from data.params import DataLimit
from debug import dbg
from ml.const import FeatureCol
from ml.dl_features import DLFeatureEngine
from ml.dl_trainer import DLTrainer
from ml.meta_learner import MetaLearner
from ml.params import SessionConfig
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer
from path import PathConfig


class QuantAIEngine:
    """
    量化 AI 引擎中樞。
    封裝了資料抓取、雙腦模型訓練 (XGBoost + DL)、Meta-Learner 融合，以及線上推論的完整管線。
    """
    def __init__(self, ticker: str):
        self.config = SessionConfig(ticker=ticker)

        # 基礎設施
        self.db = DataManager()
        self.fetcher = Fetcher()

        # 模型實體 (推論時才會載入)
        self.xgb_model = None
        self.dl_model = None
        self.meta_learner = None
        # DL 推論的縮放器
        self.dl_scaler = None

        self.scaler_path = PathConfig.get_dl_scalar_path(self.config.ticker, self.config.rnn_type)

    # ==========================================
    # 模組 1：資料更新
    # ==========================================
    def update_market_data(self, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR) -> bool:
        """供 UI 觸發：從網路抓取最新歷史資料並寫入資料庫"""
        dbg.log(f"[{self.config.ticker}] 正在從網路更新歷史資料...")
        daily_df = self.fetcher.fetch_daily_data(self.config.ticker, period=period, unit=unit)

        if not daily_df.empty:
            self.db.save_daily_data(self.config.ticker, daily_df)
            dbg.log(f"[{self.config.ticker}] 資料庫更新成功！")
            return True
        else:
            dbg.error(f"[{self.config.ticker}] 抓取資料失敗，請檢查網路。")
            return False

    # ==========================================
    # 模組 2：自動化訓練與存檔
    # ==========================================
    def train_all_models(self, save_models: bool = True):
        """
        供 UI 或開發者觸發：執行完整的 Stacking 訓練管線。
        :param save_models: 若為 True，則在 CV 驗證後，用全量資料重新訓練並儲存最終上線模型 (.pth, .json, .joblib)
        """
        dbg.log(f"🚀 開始執行 {self.config.ticker} 混合模型自動化訓練管線")

        # 取得資料
        df_raw = self.db.get_daily_data(self.config.ticker)
        if df_raw.empty:
            dbg.error(f"資料庫中無 {self.config.ticker} 的資料，請先執行 update_market_data()。")
            return

        # XGBoost 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 左腦：XGBoost ---")
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_raw, self.config.lookahead)

        xgb_trainer = XGBTrainer(self.config.ticker)
        oof_xgb = xgb_trainer.train_with_cv(df_xgb_clean)
        y_true = df_xgb_clean[FeatureCol.TARGET]

        if save_models:
            xgb_trainer.train_and_save_final_model(df_xgb_clean)

        # DL (CNN-RNN) 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 右腦：Deep Learning ---")

        dl_engine = DLFeatureEngine(self.config.lookahead)
        X_dl, y_dl, scaler, valid_index = dl_engine.process_pipeline(df_raw)

        dl_trainer = DLTrainer(ticker=self.config.ticker, rnn_type=self.config.rnn_type)
        oof_dl = dl_trainer.train_with_cv(X_dl, y_dl, valid_index)

        if save_models:
            dl_trainer.train_and_save_final_model(X_dl, y_dl)
            joblib.dump(scaler, self.scaler_path)
            dbg.log(f"DL Scaler 已儲存至: {self.scaler_path}")

        # 清理 GPU 記憶體
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        # Meta-Learner 處理管線 (Level 1)
        dbg.log("\n--- [訓練階段] 總指揮：Meta-Learner ---")
        meta_learner = MetaLearner(ticker=self.config.ticker)
        X_meta, y_meta = meta_learner.evaluate_oof(oof_xgb, oof_dl, y_true)

        if save_models:
            meta_learner.train_and_save_final_model(X_meta, y_meta)

        gc.collect()
        dbg.log(f"\n🎉 {self.config.ticker} Stacking 整合訓練管線全數執行完畢！")

    # ==========================================
    # 模組 3：載入模型 (為線上推論做準備)
    # ==========================================
    def load_inference_models(self) -> bool:
        """
        供 UI 啟動時觸發：將儲存在硬碟的權重檔載入至記憶體中。
        """
        dbg.log(f"[{self.config.ticker}] 準備載入線上推論模型...")
        try:
            xgb_path = PathConfig.get_xgboost_model_path(self.config.ticker)
            self.xgb_model = XGBTrainer.load_inference_model(xgb_path)

            self.dl_scaler = joblib.load(self.scaler_path)
            dl_input_size = len(StockCol.get_ohlcv())
            self.dl_model = DLTrainer(self.config.ticker, self.config.rnn_type).load_inference_model(dl_input_size)

            meta = MetaLearner(self.config.ticker)
            meta.load_inference_model()
            self.meta_learner = meta

            if None in (self.xgb_model, self.dl_model, self.dl_scaler, self.meta_learner.model):
                raise ValueError("部分模型或 Scaler 回傳為 None")

            dbg.log("✅ 三大模型與 Scaler 載入成功，系統已就緒！")
            return True

        except Exception as e:
            dbg.error(f"模型載入失敗，請確認是否已經執行過訓練管線: {e}")
            return False

    def predict_today(self) -> float | None:
        """
        供 UI 或行為樹呼叫：預測今天的最終勝率。
        (請確保已經呼叫過 load_inference_models)
        """
        if None in (self.xgb_model, self.dl_model, self.meta_learner, self.dl_scaler):
            dbg.error("模型未完全載入，無法進行推論！")
            return None

        # 抓取最新資料 (確保資料庫有最新 K 線)
        df_raw = self.db.get_daily_data(self.config.ticker)
        df_recent = df_raw.tail(500).copy()

        # ==========================================
        # 🟢 左腦 (XGBoost) 推論修正
        # ==========================================
        xgb_engine = XGBFeatureEngine()
        df_xgb_features = xgb_engine._create_daily_features(df_recent)
        latest_xgb_features = df_xgb_features[FeatureCol.get_features()].iloc[-1:]

        if latest_xgb_features.isna().any().any():
            dbg.war(f"[{self.config.ticker}] 警告：今日特徵包含 NaN (可能歷史資料不足 240 天)，XGB 預測準確度可能下降。")

        prob_xgb = self.xgb_model.predict_proba(latest_xgb_features)[0, 1]

        # ==========================================
        # 🟢 右腦 (DL) 推論修正
        # ==========================================
        dl_engine = DLFeatureEngine(self.config.lookahead)
        # 注意：傳入 self.dl_scaler，讓它進入「推論模式」
        X_dl, _, _, _ = dl_engine.process_pipeline(df_recent, scaler=self.dl_scaler)

        if X_dl is None:
            dbg.error(f"[{self.config.ticker}] 資料量不足，無法產生 DL 推論特徵。")
            return None

        # 轉成 Tensor 丟給模型
        self.dl_model.eval()
        with torch.no_grad():
            device = next(self.dl_model.parameters()).device
            X_tensor = torch.as_tensor(X_dl, dtype=torch.float32, device=device)
            prob_dl = torch.sigmoid(self.dl_model(X_tensor)).item()

        # 總指揮 (Meta-Learner) 融合
        final_prob = self.meta_learner.predict_final_probability(prob_xgb, prob_dl)

        dbg.log(f"[{self.config.ticker} 今日預測] XGB: {prob_xgb:.2f} | DL: {prob_dl:.2f} ➔ 最終勝率: {final_prob:.2f}")
        return final_prob
