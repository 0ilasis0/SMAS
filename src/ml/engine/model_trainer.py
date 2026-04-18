import gc
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import torch

from data.const import MacroTicker
from debug import dbg
from ml.const import FeatureCol, ModelCol, SignalCol
from ml.data.dl_features import DLFeatureEngine
from ml.data.market_features import MarketFeatureEngine
from ml.data.xgb_features import XGBFeatureEngine
from ml.model.meta_learner import MetaLearner
from ml.trainers.dl_trainer import DLTrainer
from ml.trainers.market_trainer import MarketTrainer
from ml.trainers.xgb_trainer import XGBTrainer

if TYPE_CHECKING:
    from .core import QuantAIEngine

class ModelTrainer:
    """
    [模組 3] 模型訓練長
    專責執行 XGBoost, Deep Learning, Market Brain 與 Meta-Learner 的完整 Stacking 訓練管線。
    """
    def __init__(self, engine):
        self.engine: "QuantAIEngine" = engine

    def _are_models_up_to_date(self) -> bool:
        """檢查所有核心模型是否在「今天」已經被訓練並存檔"""
        today = datetime.now().date()

        model_paths_to_check = [
            Path(self.engine.paths[ModelCol.XGB]),
            Path(self.engine.paths[ModelCol.DL]),
            Path(self.engine.paths[ModelCol.DL_SCALAR]),
            Path(self.engine.paths[ModelCol.META]),
            Path(self.engine.paths[ModelCol.MARKET])
        ]

        for p in model_paths_to_check:
            if not p.exists():
                return False
            last_modified_date = datetime.fromtimestamp(p.stat().st_mtime).date()
            if last_modified_date != today:
                return False
        return True

    def train_all_models(self, save_models: bool = True):
        """執行完整的 Stacking 訓練管線"""
        config = self.engine.config
        paths = self.engine.paths
        oos_days = self.engine.oos_days

        if save_models and self._are_models_up_to_date():
            dbg.log(f"⚡ [{config.ticker}] 系統偵測到所有模型均已是今日最新版本，跳過重複訓練！")
            return

        dbg.log(f"開始執行 {config.ticker} 訓練管線 (保留 {oos_days} 天做為純淨測試集)")

        self.engine.run_data_watchdog(config.ticker)

        macro_tickers = [e.value for e in MacroTicker]
        df_raw_full = self.engine.db.get_aligned_market_data(config.ticker, macro_tickers)

        if df_raw_full.empty:
            dbg.error(f"資料庫中無 {config.ticker} 的資料，請先執行 update_market_data()。")
            return

        df_train_only = df_raw_full.iloc[:-oos_days] if oos_days > 0 else df_raw_full

        # ================= 左腦：XGBoost =================
        dbg.log("\n--- [訓練階段] 左腦：XGBoost ---")
        xgb_engine = XGBFeatureEngine()
        df_xgb_train = xgb_engine.process_pipeline(df_train_only, config.lookahead, is_training=True)
        xgb_trainer = XGBTrainer(config.ticker)
        xgb_trainer.model_save_path = paths[ModelCol.XGB]
        oof_xgb = xgb_trainer.train_with_cv(df_xgb_train, lookahead=config.lookahead)
        if save_models:
            xgb_trainer.train_and_save_final_model(df_xgb_train, paths[ModelCol.XGB])

        y_true = df_xgb_train[FeatureCol.TARGET.value]

        # ================= 右腦：Deep Learning =================
        dbg.log("\n--- [訓練階段] 右腦：Deep Learning ---")
        dl_engine = DLFeatureEngine(config.lookahead)
        X_dl_train, y_dl_train, valid_index_train = dl_engine.process_pipeline(df_train_only, is_training=True)

        dl_trainer = DLTrainer(ticker=config.ticker, dl_model_type=config.dl_model_type, rnn_type=config.rnn_type)
        oof_dl = dl_trainer.train_with_cv(X_dl_train, y_dl_train, valid_index_train, lookahead=config.lookahead)

        if save_models:
            final_dl_scaler = dl_trainer.train_and_save_final_model(X_dl_train, y_dl_train, paths[ModelCol.DL])
            scaler_path_obj = Path(paths[ModelCol.DL_SCALAR])
            scaler_path_obj.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_dl_scaler, str(scaler_path_obj))
            dbg.log(f"DL Scaler 已儲存至: {scaler_path_obj}")

        # ================= 第三腦：Market Regime =================
        dbg.log("\n--- [訓練階段] 第三腦：Market Regime ---")
        market_path = paths[ModelCol.MARKET]

        # 檢查大盤模型是否存在，且是否為「今天」訓練的
        is_market_trained_today = False
        market_path_obj = Path(market_path)
        if market_path_obj.exists():
            mtime = datetime.fromtimestamp(market_path_obj.stat().st_mtime).date()
            if mtime == datetime.now().date():
                is_market_trained_today = True

        if not is_market_trained_today:
            dbg.log(f"未發現今日最新 (OOS={oos_days}) 的大盤防禦模型，開始進行全局訓練...")
            aux_macros = MacroTicker.get_auxiliary_tickers()
            df_market_pure = self.engine.db.get_aligned_market_data(MacroTicker.TWII.value, aux_macros)

            if df_market_pure.empty:
                dbg.error(f"🚨 找不到大盤資料 {MacroTicker.TWII.value}！請先執行 update_market_data()！")
                return

            df_market_pure_train = df_market_pure.iloc[:-oos_days] if oos_days > 0 else df_market_pure
            market_engine_feat = MarketFeatureEngine(lookahead=config.lookahead)
            df_market_train = market_engine_feat.process_pipeline(df_market_pure_train, is_training=True)

            market_trainer = MarketTrainer()
            market_trainer.train_with_cv(df_market_train, lookahead=config.lookahead)

            if save_models:
                market_trainer.train_and_save_final_model(df_market_train, market_path)
        else:
            # 現在只有真正是今天訓練的，才會印出這行跳過
            dbg.log(f"大盤防禦模型 (OOS={oos_days}) 今日已訓練更新，跳過重複訓練。")

        # 清理 GPU 記憶體
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

        # ================= 總指揮：Meta-Learner =================
        dbg.log("\n--- [訓練階段] 總指揮：Meta-Learner ---")
        df_oof = pd.DataFrame(index=df_train_only.index)

        if isinstance(oof_xgb, np.ndarray):
            oof_xgb = pd.Series(oof_xgb, index=df_xgb_train.index)
        if isinstance(oof_dl, np.ndarray):
            oof_dl = pd.Series(oof_dl, index=valid_index_train)

        df_oof = df_oof.join(oof_xgb.rename(SignalCol.PROB_XGB.value)) \
                       .join(oof_dl.rename(SignalCol.PROB_DL.value)) \
                       .join(y_true.rename(FeatureCol.TARGET.value))

        df_oof.dropna(subset=[SignalCol.PROB_XGB.value, SignalCol.PROB_DL.value, FeatureCol.TARGET.value], inplace=True)

        if df_oof.empty:
            dbg.error("OOF 對齊後資料為空！請檢查各模型的訓練資料長度。")
            return

        meta_learner = MetaLearner(ticker=config.ticker)
        X_meta, y_meta = meta_learner.evaluate_oof(df_oof[SignalCol.PROB_XGB.value], df_oof[SignalCol.PROB_DL.value], df_oof[FeatureCol.TARGET.value])

        if save_models:
            meta_learner.train_and_save_final_model(X_meta, y_meta, paths[ModelCol.META])

        gc.collect()
        dbg.log(f"\n🎉 雙層架構與三大腦模型訓練完畢！(完美避開最後 {oos_days} 天的資料)")