from pathlib import Path

import pandas as pd

from data.const import TimeUnit
from data.fetcher import Fetcher
from data.manager import DataManager
from data.params import DataLimit
from ml.const import DLModelType, ModelCol, RNNType, TradingMode
from ml.model.llm_oracle import GeminiOracle
from ml.params import SessionConfig
from path import PathConfig

from .data_updater import DataUpdater
from .data_watchdog import DataWatchdog
from .model_predictor import ModelPredictor
from .model_trainer import ModelTrainer


class QuantAIEngine:
    """
    量化 AI 引擎中樞 (模組化重構版)。
    統一對外提供 API，內部將工作委派給四大專業子模組：
    1. DataUpdater: 負責更新市場資料
    2. DataWatchdog: 負責資料除錯與平滑修復
    3. ModelTrainer: 負責四大模型訓練管線
    4. ModelPredictor: 負責模型載入與線上推論/回測
    """
    def __init__(self, ticker: str, oos_days: int,
            dl_model_type: DLModelType | None = None, rnn_type: RNNType | None = None,
            api_keys: list[str] = None
        ):

        # ================= 1. 核心配置與狀態 =================
        self.config = SessionConfig(ticker=ticker)
        if dl_model_type is not None: self.config.dl_model_type = dl_model_type
        if rnn_type is not None: self.config.rnn_type = rnn_type
        self.oos_days = oos_days

        # 模型實體 (推論時才會載入，存在記憶體中供子模組共用)
        self.xgb_model = None
        self.dl_model = None
        self.dl_scaler = None
        self.meta_learner = None
        self.market_model = None

        # 定義模型儲存路徑
        self.paths = {
            ModelCol.XGB: str(PathConfig.get_xgboost_model_path(self.config.ticker, self.oos_days)),
            ModelCol.DL: str(PathConfig.get_dl_model_path(self.config.ticker, self.config.dl_model_type, self.config.rnn_type, self.oos_days)),
            ModelCol.META: str(PathConfig.get_meta_model_path(self.config.ticker, self.oos_days)),
            ModelCol.MARKET: str(PathConfig.get_market_model_path(self.oos_days)),
            ModelCol.DL_SCALAR: str(PathConfig.get_dl_scalar_path(self.config.ticker, self.config.dl_model_type, self.config.rnn_type, self.oos_days))
        }

        # ================= 2. 基礎設施 =================
        self.db = DataManager()
        self.fetcher = Fetcher()
        self.oracle = GeminiOracle(api_keys=api_keys) if api_keys else None
        self.cache_file = Path(PathConfig.CACHE_FILE)

        # ================= 3. 實例化四大專業子模組 =================
        # 將自己 (self) 當作 Context 傳入，讓子模組可以共用 db, paths 與載入的模型
        self._updater = DataUpdater(self)
        self._watchdog = DataWatchdog(self)
        self._trainer = ModelTrainer(self)
        self._predictor = ModelPredictor(self)


    # =========================================================================
    # 公開 API (Facade 面板) - 這些方法名稱與參數必須與舊版 100% 相同
    # =========================================================================

    def update_market_data(self, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR, force_wipe: bool = False, force_sync: bool = False) -> bool:
        """模組 1：資料更新 (委派給 DataUpdater)"""
        return self._updater.update_market_data(period, unit, force_wipe, force_sync)

    def run_data_watchdog(self, ticker: str):
        """模組 2：資料防護與修復 (委派給 DataWatchdog)"""
        self._watchdog.run_data_watchdog(ticker)

    def train_all_models(self, save_models: bool = True):
        """模組 3：自動化訓練與存檔 (委派給 ModelTrainer)"""
        self._trainer.train_all_models(save_models)

    def load_inference_models(self) -> bool:
        """模組 4-1：載入模型至記憶體 (委派給 ModelPredictor)"""
        return self._predictor.load_inference_models()

    def predict_today(self, mode: TradingMode = TradingMode.SWING, is_t_minus_1_sim: bool = False) -> dict | None:
        """模組 4-2：線上推論與預測 (委派給 ModelPredictor)"""
        return self._predictor.predict_today(mode, is_t_minus_1_sim)

    def generate_backtest_data(self) -> pd.DataFrame:
        """模組 4-3：批次產生回測資料 (委派給 ModelPredictor)"""
        return self._predictor.generate_backtest_data()
