from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import pandas as pd
import torch

from base import MLTool
from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import (FeatureCol, MLConst, ModelAttr, ModelCol, OracleCol,
                      QuoteCol, SignalCol, TradingMode)
from ml.data.dl_features import DLFeatureEngine
from ml.data.market_features import MarketFeatureCol, MarketFeatureEngine
from ml.data.xgb_features import XGBFeatureEngine
from ml.model.meta_learner import MetaLearner
from ml.params import DLHyperParams
from ml.trainers.dl_trainer import DLTrainer
from ml.trainers.market_trainer import MarketTrainer
from ml.trainers.xgb_trainer import XGBTrainer

from ._params import CircuitBreakerConfig

if TYPE_CHECKING:
    from .core import QuantAIEngine

class ModelPredictor:
    """
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

            dl_input_size = len(FeatureCol.get_dl_features())
            engine.dl_model = DLTrainer(engine.config.ticker, engine.config.dl_model_type, engine.config.rnn_type).load_inference_model(dl_input_size, engine.paths[ModelCol.DL])

            engine.meta_learner = MetaLearner(engine.config.ticker)
            engine.meta_learner.load_inference_model(engine.paths[ModelCol.META])

            engine.market_model = MarketTrainer.load_inference_model(engine.paths[ModelCol.MARKET])
            engine.config.dynamic_market_threshold = engine.market_model.dynamic_threshold

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
        prob_xgb = engine.xgb_model.predict_proba(df_xgb_clean.loc[[target_date], FeatureCol.get_xgb_features()])[0, 1]

        # 右腦推論
        dl_engine = DLFeatureEngine(config.lookahead)
        X_dl_raw, _, valid_index = dl_engine.process_pipeline(df_recent, is_training=False)
        if target_date not in valid_index:
            dbg.error(f"[{config.ticker}] DL 缺失特徵，無法預測！")
            return None

        target_idx = list(valid_index).index(target_date)
        X_dl_2d = X_dl_raw[[target_idx]].reshape(-1, X_dl_raw.shape[2])
        X_dl_scaled = engine.dl_scaler.transform(X_dl_2d).reshape(1, DLHyperParams.time_steps, X_dl_raw.shape[2])

        engine.dl_model.eval()
        with torch.no_grad():
            device = next(engine.dl_model.parameters()).device
            prob_dl = torch.sigmoid(engine.dl_model(torch.as_tensor(X_dl_scaled, dtype=torch.float32, device=device))).item()

        # 第三腦推論
        market_engine_feat = MarketFeatureEngine(lookahead=config.lookahead)

        # 動態取得所有需要的總經輔助標的 (排除 TWII 自己)
        aux_macros = MacroTicker.get_auxiliary_tickers()
        df_market_pure = engine.db.get_aligned_market_data(MacroTicker.TWII.value, aux_macros).tail(MLConst.MAX_LOOKBACK)
        df_market_clean = market_engine_feat.process_pipeline(df_market_pure, is_training=False)

        if target_date not in df_market_clean.index:
            dbg.error(f"大盤缺失特徵！拒絕預測。")
            return None

        # 1. 取得模型吐出的「膨脹機率」
        market_features = MarketFeatureCol.get_features()
        raw_prob_danger_array = engine.market_model.predict_proba(
            df_market_clean[market_features].astype(float).values
        )[:, 1]

        # 2. 取出藏在模型基因裡的「還原金鑰 (權重)」
        weight = getattr(engine.market_model, ModelAttr.TRAIN_SCALE_WEIGHT, 1.0)

        # 3. 陣列化數學還原，並打包成完整的 pandas Series 時間序列
        prob_danger_array = MLTool.unscale_probability(raw_prob_danger_array, float(weight))
        # 建立臨時的 Series 以利計算移動平均
        prob_market_safe_series = pd.Series(
            1.0 - prob_danger_array,
            index=df_market_clean.index
        )

        # 4. 獨立防禦模組介入：先進行 3 日 EMA 平滑，再送入斷路器審查
        prob_market_safe_series = prob_market_safe_series.ewm(span=CircuitBreakerConfig.SMOOTH_SPAN, adjust=False).mean()
        prob_market_safe_series = self._apply_circuit_breaker(df_market_clean, prob_market_safe_series)

        # 5. 終端解鎖：精準提取出「今天 (最後一筆)」的平滑安全機率純量
        prob_market_safe = float(prob_market_safe_series.iloc[-1])

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
            FeatureCol.ATR_RATIO.value: float(df_xgb_clean[FeatureCol.ATR_RATIO].iloc[-1]) if not pd.isna(df_xgb_clean[FeatureCol.ATR_RATIO].iloc[-1]) else 0.0,
        }

    def _apply_circuit_breaker(self, df_market_clean: pd.DataFrame, prob_market_safe_series: pd.Series) -> pd.Series:
        """
        [黑天鵝防禦模組] 終極防爆斷路器
        直接抓原始收盤價現場計算，絕不引發 KeyError。
        """
        if df_market_clean.empty or prob_market_safe_series.empty:
            return prob_market_safe_series

        if len(df_market_clean) > 1:
            dbg.error(f"len(df_market_clean) = {len(df_market_clean)} > 1")
            return prob_market_safe_series

        sox_crash = False
        vix_panic = False

        # ====================================================================
        # 費半 (SOX) 暴跌檢查
        # ====================================================================
        if MarketFeatureCol.SOX_CLOSE in df_market_clean.columns and len(df_market_clean) > 1:
            sox_valid = df_market_clean[MarketFeatureCol.SOX_CLOSE].dropna()
            if len(sox_valid) > 1:
                sox_ret = sox_valid.pct_change().iloc[-1]
                sox_crash = float(sox_ret) < CircuitBreakerConfig.SOX_CRASH_THRESHOLD
        else:
            dbg.error(f"{MarketFeatureCol.SOX_CLOSE}不在{df_market_clean.columns}")

        # ====================================================================
        # 恐慌指數 (VIX) 飆升檢查
        # ====================================================================
        if MarketFeatureCol.VIX_CLOSE in df_market_clean.columns and len(df_market_clean) > 1:
            vix_valid = df_market_clean[MarketFeatureCol.VIX_CLOSE].dropna()
            if len(vix_valid) > 1:
                vix_ret = vix_valid.pct_change().iloc[-1]
                vix_panic = float(vix_ret) > CircuitBreakerConfig.VIX_PANIC_THRESHOLD
        else:
            dbg.error(f"{MarketFeatureCol.VIX_CLOSE}不在{df_market_clean.columns}")

        # ====================================================================
        # 斷路器一錘定音
        # ====================================================================
        if sox_crash or vix_panic:
            prob_market_safe_series.iloc[-1] = 0.0

            trigger_reasons = []
            if sox_crash: trigger_reasons.append("費半崩跌")
            if vix_panic: trigger_reasons.append("VIX失控")
            dbg.war(f"🚨 [核彈級警報] 觸發斷路器 ({' / '.join(trigger_reasons)})！大盤安全度強行歸零！")

        return prob_market_safe_series

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
        prob_xgb_series = pd.Series(engine.xgb_model.predict_proba(df_xgb_clean[FeatureCol.get_xgb_features()])[:, 1], index=df_xgb_clean.index, name=SignalCol.PROB_XGB.value)

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
        aux_macros = MacroTicker.get_auxiliary_tickers()
        df_market_pure = engine.db.get_aligned_market_data(MacroTicker.TWII.value, aux_macros)
        df_market_clean = market_engine_feat.process_pipeline(df_market_pure, is_training=False)

        # 1. 批次取得所有歷史日期的「膨脹機率陣列」
        raw_prob_danger_array = engine.market_model.predict_proba(df_market_clean[MarketFeatureCol.get_features()])[:, 1]

        # 2. 取出權重金鑰
        weight = getattr(engine.market_model, ModelAttr.TRAIN_SCALE_WEIGHT.value, 1.0)

        # 3. 陣列化數學還原
        prob_danger_array = MLTool.unscale_probability(raw_prob_danger_array, float(weight))

        # 4. 轉換為安全機率並打包成 Series
        prob_market_safe_series = pd.Series(
            1.0 - prob_danger_array,
            index=df_market_clean.index,
            name=SignalCol.PROB_MARKET_SAFE.value
        )
        # 使用 3 日指數移動平均 (EMA)。
        # 這意味著：必須連續幾天模型都看壞，安全機率才會真正掉下去。
        # 單一天的突發性恐慌預測會被平均掉，大幅降低交易成本與誤判率。
        smooth_span = 3
        prob_market_safe_series = prob_market_safe_series.ewm(span=smooth_span, adjust=False).mean()

        df_backtest = df_raw.copy().join(prob_xgb_series).join(prob_dl_series).join(prob_market_safe_series)
        df_backtest.dropna(subset=[SignalCol.PROB_XGB.value, SignalCol.PROB_DL.value, SignalCol.PROB_MARKET_SAFE.value], inplace=True)

        if df_backtest.empty: return pd.DataFrame()

        df_backtest[SignalCol.PROB_FINAL.value] = engine.meta_learner.model.predict_proba(df_backtest[[SignalCol.PROB_XGB.value, SignalCol.PROB_DL.value]])[:, 1]

        dbg.log(f"✅ 回測資料生成完畢！共產出 {len(df_backtest)} 筆有效預測日。")
        return df_backtest