import gc

import joblib
import numpy as np
import pandas as pd
import torch

from data.const import MacroTicker, TimeUnit
from data.fetcher import Fetcher
from data.manager import DataManager
from data.params import DataLimit
from debug import dbg
from ml.const import FeatureCol, MetaCol, MLCol, MLConst
from ml.data.dl_features import DLFeatureEngine
from ml.data.market_features import MarketFeatureCol, MarketFeatureEngine
from ml.data.xgb_features import XGBFeatureEngine
from ml.model.llm_oracle import GeminiOracle, TradingMode
from ml.model.meta_learner import MetaLearner
from ml.params import DLHyperParams, SessionConfig
from ml.trainers.dl_trainer import DLTrainer
from ml.trainers.market_trainer import MarketTrainer
from ml.trainers.xgb_trainer import XGBTrainer
from path import PathConfig


class QuantAIEngine:
    """
    量化 AI 引擎中樞。
    封裝了資料抓取、三腦模型訓練 (XGBoost + DL + Market)、Meta-Learner 融合，以及線上推論的完整管線。
    """
    def __init__(self, ticker: str, oos_days: int = 0, api_keys: list[str] = None):
        self.config = SessionConfig(ticker=ticker)
        self.oos_days = oos_days

        # 基礎設施
        self.db = DataManager()
        self.fetcher = Fetcher()

        # 模型實體 (推論時才會載入)
        self.xgb_model = None
        self.dl_model = None
        self.meta_learner = None
        self.market_model = None

        if api_keys:
            self.oracle = GeminiOracle(api_keys=api_keys, mode=TradingMode.SWING)
        else:
            self.oracle = None

        self.paths = {
            MLCol.XGB: PathConfig.get_xgboost_model_path(self.config.ticker, self.oos_days),
            MLCol.DL: PathConfig.get_dl_model_path(self.config.ticker, self.config.rnn_type, self.oos_days),
            MLCol.META: PathConfig.get_meta_model_path(self.config.ticker, self.oos_days),
            MLCol.MARKET: PathConfig.get_market_model_path(self.oos_days)
        }

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
            dbg.log(f"[{macro_ticker.value}] 正在同步更新大盤/總經資料...")
            df_macro = self.fetcher.fetch_daily_data(macro_ticker.value, period=period, unit=unit)
            if not df_macro.empty:
                self.db.save_daily_data(macro_ticker.value, df_macro)
            else:
                dbg.war(f"[{macro_ticker.value}] 總經資料更新失敗，可能被 Yahoo 阻擋或無數據。")

        return success

    # ==========================================
    # 模組 2：自動化訓練與存檔
    # ==========================================
    def train_all_models(self, save_models: bool = True,):
        """
        供 UI 或開發者觸發：執行完整的 Stacking 訓練管線。
        """
        dbg.log(f"🚀 開始執行 {self.config.ticker} 訓練管線 (保留 {self.oos_days} 天做為純淨測試集)")

        # 取得「已對齊大盤特徵」的完整資料集
        macro_tickers = [e.value for e in MacroTicker]
        df_raw_full = self.db.get_aligned_market_data(self.config.ticker, macro_tickers)

        if df_raw_full.empty:
            dbg.error(f"資料庫中無 {self.config.ticker} 的資料，請先執行 update_market_data()。")
            return

        df_train_only = df_raw_full.iloc[:-self.oos_days] if self.oos_days > 0 else df_raw_full

        # XGBoost 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 左腦：XGBoost ---")
        xgb_engine = XGBFeatureEngine()
        df_xgb_train = xgb_engine.process_pipeline(df_train_only, self.config.lookahead)
        xgb_trainer = XGBTrainer(self.config.ticker)
        xgb_trainer.model_save_path = self.paths[MLCol.XGB]
        oof_xgb = xgb_trainer.train_with_cv(df_xgb_train, lookahead=self.config.lookahead)
        if save_models:
            xgb_trainer.train_and_save_final_model(df_xgb_train, self.paths[MLCol.XGB])

        y_true = df_xgb_train[FeatureCol.TARGET]

        # DL (CNN-RNN) 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 右腦：Deep Learning ---")
        dl_engine = DLFeatureEngine(self.config.lookahead)

        X_dl_train, y_dl_train, valid_index_train = dl_engine.process_pipeline(df_train_only, is_training=True)

        dl_trainer = DLTrainer(ticker=self.config.ticker, rnn_type=self.config.rnn_type)
        oof_dl = dl_trainer.train_with_cv(X_dl_train, y_dl_train, valid_index_train, lookahead=self.config.lookahead)

        if save_models:
            final_dl_scaler = dl_trainer.train_and_save_final_model(X_dl_train, y_dl_train, self.paths[MLCol.DL])

            scaler_save_path = PathConfig.get_dl_scalar_path(self.config.ticker, self.config.rnn_type, self.oos_days)
            joblib.dump(final_dl_scaler, scaler_save_path)
            dbg.log(f"DL Scaler 已儲存至: {scaler_save_path}")

        # Market Brain 大盤防禦處理管線
        dbg.log("\n--- [訓練階段] 第三腦：Market Regime ---")

        if not self.paths[MLCol.MARKET].exists():
            dbg.log(f"未發現 OOS={self.oos_days} 的大盤防禦模型，開始進行全局訓練...")

            # 使用 Enum 避免字串打錯
            twii = MacroTicker.TWII.value
            sox = MacroTicker.SOX.value

            # 給第三腦專屬的「純淨大盤資料」，以 ^TWII 為主體！
            df_market_pure = self.db.get_aligned_market_data(twii, [sox])

            if df_market_pure.empty:
                dbg.error(f"🚨 資料庫找不到大盤資料 {twii}！請務必先執行 ai_engine.update_market_data()！")
                return

            # 確保訓練時也避開 oos_days
            df_market_pure_train = df_market_pure.iloc[:-self.oos_days] if self.oos_days > 0 else df_market_pure

            market_engine = MarketFeatureEngine(lookahead=self.config.lookahead)
            df_market_train = market_engine.process_pipeline(df_market_pure_train, is_training=True)

            # 將專屬路徑傳給 Trainer
            market_trainer = MarketTrainer()
            market_trainer.train_with_cv(df_market_train, lookahead=self.config.lookahead)

            if save_models:
                market_trainer.train_and_save_final_model(df_market_train, self.paths[MLCol.MARKET])

        else:
            dbg.log(f"大盤防禦模型 (OOS={self.oos_days}) 已存在，跳過重複訓練。")

        # 清理 GPU 記憶體
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        # Meta-Learner 處理管線 (Level 1)
        dbg.log("\n--- [訓練階段] 總指揮：Meta-Learner ---")

        # 確保 Meta-Learner 吃到的 XGB 預測、DL 預測與真實標籤，都在同一個日期上！
        df_oof = pd.DataFrame(index=df_train_only.index)

        if isinstance(oof_xgb, np.ndarray):
            oof_xgb = pd.Series(oof_xgb, index=df_xgb_train.index)
        if isinstance(oof_dl, np.ndarray):
            oof_dl = pd.Series(oof_dl, index=valid_index_train)

        df_oof = df_oof.join(oof_xgb.rename(MetaCol.PROB_XGB)) \
                       .join(oof_dl.rename(MetaCol.PROB_DL)) \
                       .join(y_true.rename(FeatureCol.TARGET))

        # 刪除長度不一致的暖機期
        df_oof.dropna(subset=[MetaCol.PROB_XGB, MetaCol.PROB_DL, FeatureCol.TARGET], inplace=True)

        if df_oof.empty:
            dbg.error("OOF 對齊後資料為空！請檢查各模型的訓練資料長度。")
            return

        aligned_oof_xgb = df_oof[MetaCol.PROB_XGB]
        aligned_oof_dl = df_oof[MetaCol.PROB_DL]
        aligned_y_true = df_oof[FeatureCol.TARGET]

        meta_learner = MetaLearner(ticker=self.config.ticker)
        X_meta, y_meta = meta_learner.evaluate_oof(aligned_oof_xgb, aligned_oof_dl, aligned_y_true)

        if save_models:
            meta_learner.train_and_save_final_model(X_meta, y_meta, self.paths[MLCol.META])

        gc.collect()
        dbg.log(f"\n🎉 雙層架構與三大腦模型訓練完畢！(完美避開最後 {self.oos_days} 天的資料)")

    # ==========================================
    # 模組 3：載入模型 (為線上推論做準備)
    # ==========================================
    def load_inference_models(self) -> bool:
        """供 UI 啟動時觸發：將儲存在硬碟的權重檔載入至記憶體中。"""
        dbg.log(f"[{self.config.ticker}] 準備載入線上推論模型 (OOS={self.oos_days})...")
        try:
            self.xgb_model = XGBTrainer.load_inference_model(self.paths[MLCol.XGB])

            scaler_path = PathConfig.get_dl_scalar_path(self.config.ticker, self.config.rnn_type, self.oos_days)
            self.dl_scaler = joblib.load(scaler_path)

            dl_input_size = DLHyperParams.INPUT_SIZE
            self.dl_model = DLTrainer(self.config.ticker, self.config.rnn_type).load_inference_model(dl_input_size, self.paths[MLCol.DL])

            self.meta_learner = MetaLearner(self.config.ticker)
            self.meta_learner.load_inference_model(self.paths[MLCol.META])

            self.market_model = MarketTrainer.load_inference_model(self.paths[MLCol.MARKET])

            if None in (self.xgb_model, self.dl_model, self.dl_scaler, self.meta_learner.model, self.market_model):
                raise ValueError("部分模型或 Scaler 回傳為 None")

            dbg.log("✅ 四大元件與 DL Scaler 載入成功，系統已就緒！")
            return True

        except Exception as e:
            dbg.error(f"模型載入失敗，請確認是否已經執行過訓練管線: {e}")
            return False

    def predict_today(self) -> dict | None:
        """
        供 UI 或行為樹呼叫：預測今天的最終勝率與大盤安全度。
        回傳: (final_prob, prob_market_safe)
        """
        if None in (self.xgb_model, self.dl_model, self.meta_learner, self.dl_scaler, self.market_model):
            dbg.error("模型未完全載入，無法進行推論！")
            return None

        # 抓取最新資料時，必須包含對齊後的大盤數據
        macro_tickers = [e.value for e in MacroTicker]
        df_raw = self.db.get_aligned_market_data(self.config.ticker, macro_tickers)
        if df_raw.empty: return None

        df_recent = df_raw.tail(MLConst.MAX_LOOKBACK).copy()
        target_date = df_recent.index[-1]
        dbg.log(f"正在預測目標日期: {target_date.strftime('%Y-%m-%d')}")

        # 左腦 (XGBoost) 推論
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_recent, self.config.lookahead, is_training=False)
        if target_date not in df_xgb_clean.index:
            dbg.error(f"[{self.config.ticker}] XGBoost 缺失 {target_date} 特徵，無法預測！")
            return None
        latest_xgb_features = df_xgb_clean.loc[[target_date], FeatureCol.get_features()]
        prob_xgb = self.xgb_model.predict_proba(latest_xgb_features)[0, 1]

        # 右腦 (DL) 推論
        dl_engine = DLFeatureEngine(self.config.lookahead)
        X_dl_raw, _, valid_index = dl_engine.process_pipeline(df_recent, is_training=False)

        if target_date not in valid_index:
            dbg.error(f"[{self.config.ticker}] DL 缺失 {target_date} 特徵，無法預測！")
            return None

        target_idx = list(valid_index).index(target_date)

        num_features = X_dl_raw.shape[2]
        X_dl_2d = X_dl_raw[[target_idx]].reshape(-1, num_features)
        X_dl_scaled = self.dl_scaler.transform(X_dl_2d).reshape(1, DLHyperParams.TIME_STEPS, num_features)

        self.dl_model.eval()
        with torch.no_grad():
            device = next(self.dl_model.parameters()).device
            X_tensor = torch.as_tensor(X_dl_scaled, dtype=torch.float32, device=device)
            prob_dl = torch.sigmoid(self.dl_model(X_tensor)).item()

        # 第三腦 (Market Brain) 推論
        market_engine = MarketFeatureEngine(lookahead=self.config.lookahead)

        df_market_pure = self.db.get_aligned_market_data('^TWII', ['^SOX']).tail(MLConst.MAX_LOOKBACK)
        df_market_clean = market_engine.process_pipeline(df_market_pure, is_training=False)

        if target_date not in df_market_clean.index:
            dbg.error(f"大盤缺失 {target_date} 特徵 (可能是總經資料 API 延遲)！拒絕預測。")
            return None

        latest_market_features = df_market_clean.loc[[target_date], MarketFeatureCol.get_features()]
        prob_danger = self.market_model.predict_proba(latest_market_features)[0, 1]
        prob_market_safe = 1.0 - prob_danger

        # 總指揮 (Meta-Learner) 融合
        final_prob = self.meta_learner.predict_final_probability(prob_xgb, prob_dl)

        #啟動 LLM (GeminiOracle) 獲取情緒分數
        sentiment_score = 5
        sentiment_reason = "未提供 API Key，略過情緒分析"

        if self.oracle:
            try:
                dbg.log("啟動 LLM 神諭機，進行即時新聞情緒掃描...")
                sentiment_score, sentiment_reason = self.oracle.get_sentiment_score(self.config.ticker)
            except Exception as e:
                dbg.war(f"神諭機執行失敗，退回中立情緒: {e}")

        dbg.log(f"[{self.config.ticker} 今日總結] 勝率: {final_prob:.2%} | 大盤安全度: {prob_market_safe:.2%} | 新聞情緒: {sentiment_score}分")

        # 將所有預測結果打包回傳
        return {
            "ticker": self.config.ticker,
            "date": target_date.strftime('%Y-%m-%d'),
            "prob_final": final_prob,
            "prob_xgb": prob_xgb,
            "prob_dl": prob_dl,
            "prob_market_safe": prob_market_safe,
            "sentiment_score": sentiment_score,
            "sentiment_reason": sentiment_reason
        }

    def generate_backtest_data(self) -> pd.DataFrame:
        """
        批次產生包含歷史 K 線、AI 預測勝率與大盤安全機率的 DataFrame。
        """
        if None in (self.xgb_model, self.dl_model, self.meta_learner, self.dl_scaler, self.market_model):
            dbg.error("模型未載入！請先執行 load_inference_models()")
            return pd.DataFrame()

        dbg.log(f"[{self.config.ticker}] 正在批次生成歷史預測勝率 (Backtest Data)...")

        # 基底改為具備大盤特徵的 DataFrame
        macro_tickers = [e.value for e in MacroTicker]
        df_raw = self.db.get_aligned_market_data(self.config.ticker, macro_tickers)

        if df_raw.empty: return pd.DataFrame()

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
        X_dl_raw, _, valid_index = dl_engine.process_pipeline(df_raw, is_training=False)

        num_features = X_dl_raw.shape[2]
        X_dl_2d = X_dl_raw.reshape(-1, num_features)
        X_dl_scaled = self.dl_scaler.transform(X_dl_2d).reshape(X_dl_raw.shape)

        self.dl_model.eval()
        with torch.no_grad():
            device = next(self.dl_model.parameters()).device
            X_tensor = torch.as_tensor(X_dl_scaled, dtype=torch.float32, device=device)
            prob_dl_array = torch.sigmoid(self.dl_model(X_tensor)).cpu().numpy().flatten()
        prob_dl_series = pd.Series(prob_dl_array, index=valid_index, name=MetaCol.PROB_DL)

        # Market 批次推論
        market_engine = MarketFeatureEngine(lookahead=self.config.lookahead)

        # 抓取純淨的大盤全量資料
        df_market_pure = self.db.get_aligned_market_data('^TWII', ['^SOX'])
        df_market_clean = market_engine.process_pipeline(df_market_pure, is_training=False)

        X_market = df_market_clean[MarketFeatureCol.get_features()]
        prob_danger_array = self.market_model.predict_proba(X_market)[:, 1]

        prob_market_safe_series = pd.Series(
            1.0 - prob_danger_array,
            index=df_market_clean.index,  # 這裡的 index 是 TWII 的日期
            name=MetaCol.PROB_MARKET_SAFE
        )

        # df_raw (個股日期) 會自動去 Left Join TWII 的日期，完美對齊！
        df_backtest = df_raw.copy()
        df_backtest = df_backtest.join(prob_xgb_series).join(prob_dl_series).join(prob_market_safe_series)

        # 清除暖機期的 NaN (只要有一顆大腦沒訊號，那天就不能做決策)
        df_backtest.dropna(subset=[MetaCol.PROB_XGB, MetaCol.PROB_DL, MetaCol.PROB_MARKET_SAFE], inplace=True)

        if df_backtest.empty:
            dbg.war("合併後的預測資料為空，請檢查資料長度是否足夠讓模型暖機。")
            return pd.DataFrame()

        # Meta 融合預測
        X_meta = df_backtest[[MetaCol.PROB_XGB, MetaCol.PROB_DL]].values
        df_backtest[MetaCol.PROB_FINAL] = self.meta_learner.model.predict_proba(X_meta)[:, 1]

        dbg.log(f"✅ 回測資料生成完畢！共產出 {len(df_backtest)} 筆有效預測日。")
        return df_backtest
