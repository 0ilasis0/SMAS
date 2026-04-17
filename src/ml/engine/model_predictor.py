from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import torch

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import (FeatureCol, MLConst, ModelCol, OracleCol, QuoteCol,
                      SignalCol, TradingMode)
from ml.data.dl_features import DLFeatureEngine
from ml.data.market_features import MarketFeatureCol, MarketFeatureEngine
from ml.data.xgb_features import XGBFeatureEngine
from ml.model.meta_learner import MetaLearner
from ml.params import DLHyperParams
from ml.trainers.dl_trainer import DLTrainer
from ml.trainers.market_trainer import MarketTrainer
from ml.trainers.xgb_trainer import XGBTrainer

if TYPE_CHECKING:
    from .core import QuantAIEngine

class ModelPredictor:
    """
    [模組 4] 模型推論官
    負責載入硬碟中的權重檔，並執行線上即時推論或回測批次預測。
    支援 T-1 時光機模式 (is_t_minus_1_sim)。
    """
    def __init__(self, engine):
        self.engine: "QuantAIEngine" = engine

    def load_inference_models(self) -> bool:
        """將儲存在硬碟的權重檔載入至 engine 實體記憶體中。"""
        engine = self.engine
        dbg.log(f"[{engine.config.ticker}] 準備載入線上推論模型 (OOS={engine.oos_days})...")
        try:
            engine.xgb_model = XGBTrainer.load_inference_model(engine.paths[ModelCol.XGB])

            scaler_path_obj = Path(engine.paths[ModelCol.DL_SCALAR])
            if not scaler_path_obj.exists():
                dbg.error(f"DL Scaler 載入失敗: 找不到檔案 {scaler_path_obj}")
                return False
            engine.dl_scaler = joblib.load(str(scaler_path_obj))

            dl_input_size = DLHyperParams.INPUT_SIZE
            engine.dl_model = DLTrainer(engine.config.ticker, engine.config.dl_model_type, engine.config.rnn_type).load_inference_model(dl_input_size, engine.paths[ModelCol.DL])

            engine.meta_learner = MetaLearner(engine.config.ticker)
            engine.meta_learner.load_inference_model(engine.paths[ModelCol.META])

            engine.market_model = MarketTrainer.load_inference_model(engine.paths[ModelCol.MARKET])

            loaded_status = {
                ModelCol.XGB: engine.xgb_model,
                ModelCol.DL: engine.dl_model,
                ModelCol.DL_SCALAR: engine.dl_scaler,
                ModelCol.META: engine.meta_learner.model,
                ModelCol.MARKET: engine.market_model
            }

            missing = [name for name, obj in loaded_status.items() if obj is None]
            if missing:
                error_msg = f"❌ 以下模型/Scaler 讀取後回傳為 None: {', '.join(missing)}"
                dbg.error(error_msg)
                dbg.war("--- 系統預期的檔案路徑如下 ---")
                for k, path_str in engine.paths.items():
                    dbg.war(f"{k.name}: {path_str}")
                raise ValueError(error_msg)

            dbg.log("✅ 四大元件與 DL Scaler 載入成功，系統已就緒！")
            return True
        except Exception as e:
            dbg.error(f"模型載入失敗，請確認是否已經執行過訓練管線: {e}")
            return False

    def predict_today(self, mode: TradingMode = TradingMode.SWING, is_t_minus_1_sim: bool = False) -> dict | None:
        """預測今天的最終勝率與大盤安全度。支援時光機退回一天。"""
        engine = self.engine
        config = engine.config

        if None in (engine.xgb_model, engine.dl_model, engine.meta_learner, engine.dl_scaler, engine.market_model):
            dbg.error("模型未完全載入，無法進行推論！")
            return None

        engine.run_data_watchdog(config.ticker)

        macro_tickers = [e.value for e in MacroTicker]
        df_raw = engine.db.get_aligned_market_data(config.ticker, macro_tickers)
        if df_raw.empty: return None

        # 先記下真實的最新價，留給 UI 算市值
        real_latest_price = float(df_raw[StockCol.CLOSE].iloc[-1])

        # 🕒 啟動時光機機制
        if is_t_minus_1_sim and len(df_raw) > 1:
            df_raw = df_raw.iloc[:-1]
            dbg.log(f"🕒 [時光機模式啟動] 系統已退回至 T-1 日，準備預測: {df_raw.index[-1].strftime('%Y-%m-%d')} 的隔日走勢。")

        df_recent = df_raw.tail(MLConst.MAX_LOOKBACK).copy()
        target_date = df_recent.index[-1]
        dbg.log(f"正在預測目標日期: {target_date.strftime('%Y-%m-%d')}")

        # 左腦推論
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_recent, config.lookahead, is_training=False)
        if target_date not in df_xgb_clean.index:
            dbg.error(f"[{config.ticker}] XGBoost 缺失特徵，無法預測！")
            return None
        prob_xgb = engine.xgb_model.predict_proba(df_xgb_clean.loc[[target_date], FeatureCol.get_features()])[0, 1]

        # 右腦推論
        dl_engine = DLFeatureEngine(config.lookahead)
        X_dl_raw, _, valid_index = dl_engine.process_pipeline(df_recent, is_training=False)
        if target_date not in valid_index:
            dbg.error(f"[{config.ticker}] DL 缺失特徵，無法預測！")
            return None

        target_idx = list(valid_index).index(target_date)
        X_dl_2d = X_dl_raw[[target_idx]].reshape(-1, X_dl_raw.shape[2])
        X_dl_scaled = engine.dl_scaler.transform(X_dl_2d).reshape(1, DLHyperParams.TIME_STEPS, X_dl_raw.shape[2])

        engine.dl_model.eval()
        with torch.no_grad():
            device = next(engine.dl_model.parameters()).device
            prob_dl = torch.sigmoid(engine.dl_model(torch.as_tensor(X_dl_scaled, dtype=torch.float32, device=device))).item()

        # 第三腦推論
        market_engine_feat = MarketFeatureEngine(lookahead=config.lookahead)
        df_market_pure = engine.db.get_aligned_market_data(MacroTicker.TWII.value, [MacroTicker.SOX.value]).tail(MLConst.MAX_LOOKBACK)
        df_market_clean = market_engine_feat.process_pipeline(df_market_pure, is_training=False)
        if target_date not in df_market_clean.index:
            dbg.error(f"大盤缺失特徵！拒絕預測。")
            return None

        prob_danger = engine.market_model.predict_proba(df_market_clean.loc[[target_date], MarketFeatureCol.get_features()].astype(float).values)[0, 1]
        prob_market_safe = 1.0 - prob_danger

        # 總指揮融合
        final_prob = engine.meta_learner.predict_final_probability(prob_xgb, prob_dl)

        sentiment_score = 5
        sentiment_reason = "未提供 API Key，略過情緒分析"
        if engine.oracle:
            try:
                engine.oracle.mode = mode
                sentiment_score, sentiment_reason = engine.oracle.get_sentiment_score(config.ticker)
            except Exception as e:
                dbg.war(f"LLM 執行失敗: {e}")

        dbg.log(f"[{config.ticker} 今日總結] 勝率: {final_prob:.2%} | 大盤: {prob_market_safe:.2%} | 情緒: {sentiment_score}分")

        return {
            QuoteCol.TICKER.value: config.ticker,
            QuoteCol.DATE.value: target_date.strftime('%Y-%m-%d'),
            SignalCol.PROB_FINAL.value: final_prob,
            SignalCol.PROB_XGB.value: prob_xgb,
            SignalCol.PROB_DL.value: prob_dl,
            SignalCol.PROB_MARKET_SAFE.value: prob_market_safe,
            OracleCol.SCORE.value: sentiment_score,
            OracleCol.REASON.value: sentiment_reason,
            QuoteCol.CURRENT_PRICE.value: float(df_recent[StockCol.CLOSE].iloc[-1]),
            QuoteCol.REAL_LATEST_PRICE.value: real_latest_price,
            QuoteCol.AVG_5D_VOL.value: float(df_recent[StockCol.VOLUME].tail(5).mean()) if not pd.isna(df_recent[StockCol.VOLUME].tail(5).mean()) else 0.0,
            FeatureCol.BIAS_MONTH.value: float(df_xgb_clean[FeatureCol.BIAS_MONTH].iloc[-1]) if not pd.isna(df_xgb_clean[FeatureCol.BIAS_MONTH].iloc[-1]) else 0.0,
            FeatureCol.RETURN_5D.value: float(df_xgb_clean[FeatureCol.RETURN_5D].iloc[-1]) if not pd.isna(df_xgb_clean[FeatureCol.RETURN_5D].iloc[-1]) else 0.0,
            FeatureCol.ATR_RATIO.value: float(df_xgb_clean[FeatureCol.ATR_RATIO].iloc[-1]) if not pd.isna(df_xgb_clean[FeatureCol.ATR_RATIO].iloc[-1]) else 0.0,
            FeatureCol.TREND_STRENGTH.value: float(df_xgb_clean[FeatureCol.TREND_STRENGTH].iloc[-1]) if not pd.isna(df_xgb_clean[FeatureCol.TREND_STRENGTH].iloc[-1]) else 0.0
        }

    def generate_backtest_data(self) -> pd.DataFrame:
        """批次產生回測資料 (包含預測勝率與大盤安全度)"""
        engine = self.engine
        config = engine.config

        if None in (engine.xgb_model, engine.dl_model, engine.meta_learner, engine.dl_scaler, engine.market_model):
            dbg.error("模型未載入！請先執行 load_inference_models()")
            return pd.DataFrame()

        dbg.log(f"[{config.ticker}] 正在批次生成歷史預測勝率 (Backtest Data)...")
        engine.run_data_watchdog(config.ticker)

        df_raw = engine.db.get_aligned_market_data(config.ticker, [e.value for e in MacroTicker])
        if df_raw.empty:
            return pd.DataFrame()

        # XGB
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_raw, config.lookahead, is_training=False)
        prob_xgb_series = pd.Series(engine.xgb_model.predict_proba(df_xgb_clean[FeatureCol.get_features()])[:, 1], index=df_xgb_clean.index, name=SignalCol.PROB_XGB.value)

        # DL
        dl_engine = DLFeatureEngine(config.lookahead)
        X_dl_raw, _, valid_index = dl_engine.process_pipeline(df_raw, is_training=False)
        X_dl_scaled = engine.dl_scaler.transform(X_dl_raw.reshape(-1, X_dl_raw.shape[2])).reshape(X_dl_raw.shape)
        engine.dl_model.eval()
        with torch.no_grad():
            device = next(engine.dl_model.parameters()).device
            prob_dl_array = torch.sigmoid(engine.dl_model(torch.as_tensor(X_dl_scaled, dtype=torch.float32, device=device))).cpu().numpy().flatten()
        prob_dl_series = pd.Series(prob_dl_array, index=valid_index, name=SignalCol.PROB_DL.value)

        # Market
        market_engine_feat = MarketFeatureEngine(lookahead=config.lookahead)
        df_market_clean = market_engine_feat.process_pipeline(engine.db.get_aligned_market_data(MacroTicker.TWII.value, [MacroTicker.SOX.value]), is_training=False)
        prob_market_safe_series = pd.Series(1.0 - engine.market_model.predict_proba(df_market_clean[MarketFeatureCol.get_features()])[:, 1], index=df_market_clean.index, name=SignalCol.PROB_MARKET_SAFE.value)

        df_backtest = df_raw.copy().join(prob_xgb_series).join(prob_dl_series).join(prob_market_safe_series)
        df_backtest.dropna(subset=[SignalCol.PROB_XGB.value, SignalCol.PROB_DL.value, SignalCol.PROB_MARKET_SAFE.value], inplace=True)

        if df_backtest.empty: return pd.DataFrame()

        df_backtest[SignalCol.PROB_FINAL.value] = engine.meta_learner.model.predict_proba(df_backtest[[SignalCol.PROB_XGB.value, SignalCol.PROB_DL.value]])[:, 1]

        dbg.log(f"✅ 回測資料生成完畢！共產出 {len(df_backtest)} 筆有效預測日。")
        return df_backtest