# dl/engine.py
import gc

import joblib
import pandas as pd
import torch

from data.const import MacroTicker, TimeUnit
from data.fetcher import Fetcher
from data.manager import DataManager
from data.params import DataLimit
from debug import dbg
from ml.const import FeatureCol, MetaCol, MLConst
from ml.dl_features import DLFeatureEngine
from ml.dl_trainer import DLTrainer
from ml.meta_learner import MetaLearner
from ml.params import DLHyperParams, SessionConfig
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
        self.market_model = None

        # DL 推論的縮放器
        self.dl_scaler = None

        self.scaler_path = PathConfig.get_dl_scalar_path(self.config.ticker, self.config.rnn_type)

    # ==========================================
    # 模組 1：資料更新
    # ==========================================
    def update_market_data(self, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR) -> bool:
        """供 UI 觸發：從網路抓取最新歷史資料並寫入資料庫"""
        dbg.log(f"[{self.config.ticker}] 正在從網路更新個股歷史資料...")
        daily_df = self.fetcher.fetch_daily_data(self.config.ticker, period=period, unit=unit)
        success = False

        if not daily_df.empty:
            self.db.save_daily_data(self.config.ticker, daily_df)
            dbg.log(f"[{self.config.ticker}] 資料庫更新成功！")
            success = True
        else:
            dbg.error(f"[{self.config.ticker}] 抓取資料失敗，請檢查網路。")

        for macro_ticker in MacroTicker:
            dbg.log(f"[{macro_ticker}] 正在同步更新大盤/總經資料...")
            df_macro = self.fetcher.fetch_daily_data(macro_ticker, period=period, unit=unit)
            if not df_macro.empty:
                self.db.save_daily_data(macro_ticker, df_macro)
            else:
                dbg.war(f"[{macro_ticker}] 總經資料更新失敗，可能被 Yahoo 阻擋或無數據。")

        return success

    # ==========================================
    # 模組 2：自動化訓練與存檔
    # ==========================================
    def train_all_models(self, save_models: bool = True, oos_days: int = 0):
        """
        供 UI 或開發者觸發：執行完整的 Stacking 訓練管線。
        :param save_models: 若為 True，則在 CV 驗證後，用全量資料重新訓練並儲存最終上線模型 (.pth, .json, .joblib)
        :param oos_days: 0 代表全部資料都使用，不然會對資料進行從今日往回推oos_days天的資料不使用
        """
        dbg.log(f"🚀 開始執行 {self.config.ticker} 訓練管線 (保留 {oos_days} 天做為純淨測試集)")

        # 取得資料
        macro_tickers = MacroTicker.get_overseas_tickers
        df_raw_full = self.db.get_aligned_market_data(self.config.ticker, macro_tickers)

        if df_raw_full.empty:
            dbg.error(f"資料庫中無資料...")
            return

        df_train_only = df_raw_full.iloc[:-oos_days] if oos_days > 0 else df_raw_full

        # XGBoost 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 左腦：XGBoost ---")

        xgb_engine = XGBFeatureEngine()
        df_xgb_train = xgb_engine.process_pipeline(df_train_only, self.config.lookahead)

        xgb_trainer = XGBTrainer(self.config.ticker)
        oof_xgb = xgb_trainer.train_with_cv(df_xgb_train)
        y_true = df_xgb_train[FeatureCol.TARGET]

        if save_models:
            xgb_trainer.train_and_save_final_model(df_xgb_train)

        # DL (CNN-RNN) 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 右腦：Deep Learning ---")

        dl_engine = DLFeatureEngine(self.config.lookahead)
        X_dl_train, y_dl_train, scaler, valid_index_train = dl_engine.process_pipeline(df_train_only)

        dl_trainer = DLTrainer(ticker=self.config.ticker, rnn_type=self.config.rnn_type)
        oof_dl = dl_trainer.train_with_cv(X_dl_train, y_dl_train, valid_index_train)

        if save_models:
            dl_trainer.train_and_save_final_model(X_dl_train, y_dl_train)
            import joblib
            joblib.dump(scaler, self.scaler_path)

        dbg.log("\n--- [訓練階段] 第三腦：Market Regime ---")

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
        dbg.log("\n🎉 模型訓練完畢！(完美避開最後 250 天的資料)")

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
            dl_input_size = DLHyperParams.INPUT_SIZE
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
        df_recent = df_raw.tail(MLConst.MAX_LOOKBACK).copy()

        # ==========================================
        # 🟢 左腦 (XGBoost) 推論修正
        # ==========================================
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_recent, self.config.lookahead, is_training=False)
        if df_xgb_clean.empty:
            dbg.error(f"[{self.config.ticker}] XGBoost 暖機資料不足，無法預測！")
            return None

        latest_xgb_features = df_xgb_clean[FeatureCol.get_features()].iloc[-1:]
        prob_xgb = self.xgb_model.predict_proba(latest_xgb_features)[0, 1]

        # ==========================================
        # 🟢 右腦 (DL) 推論修正
        # ==========================================
        dl_engine = DLFeatureEngine(self.config.lookahead)

        X_dl, _, _, _ = dl_engine.process_pipeline(df_recent, scaler=self.dl_scaler)

        if X_dl is None:
            dbg.error(f"[{self.config.ticker}] 資料量不足，無法產生 DL 推論特徵。")
            return None

        # 轉成 Tensor 丟給模型
        self.dl_model.eval()
        with torch.no_grad():
            device = next(self.dl_model.parameters()).device
            X_tensor = torch.as_tensor(X_dl[-1:], dtype=torch.float32, device=device)
            prob_dl = torch.sigmoid(self.dl_model(X_tensor)).item()

        # 總指揮 (Meta-Learner) 融合
        final_prob = self.meta_learner.predict_final_probability(prob_xgb, prob_dl)

        dbg.log(f"[{self.config.ticker} 今日預測] XGB: {prob_xgb:.2f} | DL: {prob_dl:.2f} ➔ 最終勝率: {final_prob:.2f}")
        return final_prob

    def generate_backtest_data(self) -> pd.DataFrame:
        """
        【供回測引擎使用】
        批次產生包含歷史 K 線與 AI 預測勝率的 DataFrame。
        """
        if None in (self.xgb_model, self.dl_model, self.meta_learner, self.dl_scaler):
            dbg.error("模型未載入！請先執行 load_inference_models()")
            return pd.DataFrame()

        dbg.log(f"[{self.config.ticker}] 正在批次生成歷史預測勝率 (Backtest Data)...")
        df_raw = self.db.get_daily_data(self.config.ticker)
        if df_raw.empty:
            return pd.DataFrame()

        # XGBoost 批次推論
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_raw, self.config.lookahead, is_training=False)
        X_xgb = df_xgb_clean[FeatureCol.get_features()]
        prob_xgb_series = pd.Series(
            self.xgb_model.predict_proba(X_xgb)[:, 1],
            index=df_xgb_clean.index,
            name=MetaCol.PROB_XGB
        )

        # DL 批次推論
        dl_engine = DLFeatureEngine(self.config.lookahead)
        X_dl, _, _, valid_index = dl_engine.process_pipeline(df_raw, scaler=self.dl_scaler)

        self.dl_model.eval()
        with torch.no_grad():
            device = next(self.dl_model.parameters()).device
            X_tensor = torch.as_tensor(X_dl, dtype=torch.float32, device=device)
            prob_dl_array = torch.sigmoid(self.dl_model(X_tensor)).cpu().numpy().flatten()

        prob_dl_series = pd.Series(prob_dl_array, index=valid_index, name=MetaCol.PROB_DL)

        df_backtest = df_raw.copy()
        df_backtest = df_backtest.join(prob_xgb_series).join(prob_dl_series)

        df_backtest.dropna(subset=[MetaCol.PROB_XGB, MetaCol.PROB_DL], inplace=True)

        if df_backtest.empty:
            dbg.war("合併後的預測資料為空，請檢查資料長度是否足夠讓模型暖機。")
            return pd.DataFrame()

        X_meta = df_backtest[[MetaCol.PROB_XGB, MetaCol.PROB_DL]].values
        df_backtest[MetaCol.PROB_FINAL] = self.meta_learner.model.predict_proba(X_meta)[:, 1]

        dbg.log(f"✅ 回測資料生成完畢！共產出 {len(df_backtest)} 筆有效預測日。")
        return df_backtest
