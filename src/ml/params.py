from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class IndicatorParams:
# 均線參數
    MA_WEEK: int = 5
    MA_MONTH: int = 20
    MA_QUARTER: int = 60
    MA_YEAR: int = 240

    # RSI 參數
    RSI_PERIOD: int = 14

    # MACD 參數
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9


class FeatureCol(StrEnum):
    # 均線特徵
    BIAS_WEEK = "bias_week"
    BIAS_MONTH = "bias_month"
    BIAS_QUARTER = "bias_quarter"
    BIAS_YEAR = "bias_year"

    # 技術指標
    RSI = "rsi"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"

    # 動能特徵
    VOL_CHANGE = "vol_change"
    CLOSE_CHANGE = "close_change"

    # 標籤 (Label)
    TARGET = "target"

    @classmethod
    def get_features(cls):
        """回傳所有要餵給 AI 學習的特徵欄位清單 (排除 Target)"""
        return [
            cls.BIAS_WEEK, cls.BIAS_MONTH, cls.BIAS_QUARTER, cls.BIAS_YEAR,
            cls.RSI, cls.MACD, cls.MACD_SIGNAL,
            cls.VOL_CHANGE, cls.CLOSE_CHANGE
        ]


@dataclass(frozen=True)
class XGBHyperParams:
    objective: str = 'binary:logistic'  # 輸出 0~1 的機率
    eval_metric: str = 'auc'            # 使用 AUC 評估模型排序能力
    max_depth: int = 6                  # 限制樹的深度，防止過度擬合 (Overfitting)
    learning_rate: float = 0.02
    n_estimators: int = 500
    subsample: float = 0.8              # 每次建樹只用 80% 的樣本 (增加泛化能力)
    colsample_bytree: float = 0.8       # 每次建樹只用 80% 的特徵
    random_state: int = 42              # 固定亂數種子，確保結果可重現

@dataclass(frozen=True)
class TrainConfig:
    N_SPLITS: int = 5
    N_SPLITS_MAX: int = 8
    N_SPLITS_MIN: int = 3

    EARLY_STOP_ROUND = 50

@dataclass(frozen=True)
class DLHyperParams:
    """CNN-LSTM 深度學習超參數"""
    INPUT_SIZE: int = 9                 # 特徵數量 (Bias*4, RSI, MACD*2, Vol_chg, Close_chg)
    TIME_STEPS: int = 10                # 滑動窗口大小 (回顧過去 ~ 天)
    CNN_OUT_CHANNELS: int = 16          # CNN 特徵提取後的維度
    LSTM_HIDDEN: int = 32               # LSTM 隱藏層神經元數量
    NUM_LAYERS: int = 1
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 50
    DROPOUT: float = 0.2
    SCHEDULER_PATIENCE: int = 3
    SCHEDULER_FACTOR: float = 0.5

class RNNType(StrEnum):
    LSTM = "LSTM"
    GRU = "GRU"
