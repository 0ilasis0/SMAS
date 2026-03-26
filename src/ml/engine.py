import torch

from data.fetcher import Fetcher
from data.manager import DataManager
from data.variable import DataLimit, TimeUnit
from debug import dbg
from ml.dl_features import DLFeatureEngine
from ml.dl_trainer import DLTrainer
from ml.meta_learner import MetaLearner
from ml.params import FeatureCol, RNNType
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer


class QuantAIEngine:
    """
    量化 AI 引擎中樞。
    封裝了資料抓取、雙腦模型訓練 (XGBoost + DL)、Meta-Learner 融合，以及線上推論的完整管線。
    """
    def __init__(self, ticker: str, rnn_type: RNNType = RNNType.GRU):
        self.ticker = ticker
        self.rnn_type = rnn_type

        # 基礎設施
        self.db = DataManager()
        self.fetcher = Fetcher()

        # 模型實體 (推論時才會載入)
        self.xgb_model = None
        self.dl_model = None
        self.meta_model = None

    # ==========================================
    # 模組 1：資料更新
    # ==========================================
    def update_market_data(self, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR) -> bool:
        """供 UI 觸發：從網路抓取最新歷史資料並寫入資料庫"""
        dbg.log(f"[{self.ticker}] 正在從網路更新歷史資料...")
        daily_df = self.fetcher.fetch_daily_data(self.ticker, period=period, unit=unit)

        if not daily_df.empty:
            self.db.save_daily_data(self.ticker, daily_df)
            dbg.log(f"[{self.ticker}] 資料庫更新成功！")
            return True
        else:
            dbg.error(f"[{self.ticker}] 抓取資料失敗，請檢查網路。")
            return False

    # ==========================================
    # 模組 2：自動化訓練與存檔
    # ==========================================
    def train_all_models(self, save_models: bool = True):
        """
        供 UI 或開發者觸發：執行完整的 Stacking 訓練管線。
        :param save_models: 若為 True，則在 CV 驗證後，用全量資料重新訓練並儲存最終上線模型 (.pth, .json, .joblib)
        """
        dbg.log(f"🚀 開始執行 {self.ticker} 混合模型自動化訓練管線")

        # 取得資料
        df_raw = self.db.get_daily_data(self.ticker)
        if df_raw.empty:
            dbg.error(f"資料庫中無 {self.ticker} 的資料，請先執行 update_market_data()。")
            return

        # XGBoost 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 左腦：XGBoost ---")
        xgb_engine = XGBFeatureEngine()
        df_xgb_clean = xgb_engine.process_pipeline(df_raw)

        xgb_trainer = XGBTrainer()
        oof_xgb = xgb_trainer.train_with_cv(df_xgb_clean)
        y_true = df_xgb_clean[FeatureCol.TARGET]

        if save_models:
            xgb_trainer.train_and_save_final_model(df_xgb_clean)

        # DL (CNN-RNN) 處理管線 (Level 0)
        dbg.log("\n--- [訓練階段] 右腦：Deep Learning ---")

        dl_engine = DLFeatureEngine()
        X_dl, y_dl, _, valid_index = dl_engine.process_pipeline(df_raw)

        dl_trainer = DLTrainer(ticker=self.ticker, rnn_type=self.rnn_type)
        oof_dl = dl_trainer.train_with_cv(X_dl, y_dl, valid_index)

        if save_models:
            dl_trainer.train_and_save_final_model(X_dl, y_dl)

        # 清理 GPU 記憶體
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

        # Meta-Learner 處理管線 (Level 1)
        dbg.log("\n--- [訓練階段] 總指揮：Meta-Learner ---")
        meta_learner = MetaLearner(ticker=self.ticker)
        meta_learner.train_and_evaluate(oof_xgb, oof_dl, y_true)

        dbg.log(f"\n🎉 {self.ticker} Stacking 整合訓練管線全數執行完畢！")

    # ==========================================
    # 模組 3：載入模型 (為線上推論做準備)
    # ==========================================
    def load_inference_models(self) -> bool:
        """
        供 UI 啟動時觸發：將儲存在硬碟的權重檔載入至記憶體中。
        """
        dbg.log(f"[{self.ticker}] 準備載入線上推論模型...")
        try:
            self.xgb_model = XGBTrainer().load_inference_model()
            self.dl_model = DLTrainer(self.ticker, self.rnn_type).load_inference_model(...)
            self.meta_model = MetaLearner(self.ticker).load_inference_model()

            dbg.log("✅ 三大模型載入成功，系統已就緒！")
            return True
        except Exception as e:
            dbg.error(f"模型載入失敗，可能尚未訓練: {e}")
            return False
