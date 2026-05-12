from dataclasses import asdict, dataclass
from typing import Any, Dict

from ml.const import DLModelType, RNNType


@dataclass
class SessionConfig:
    """存放當前任務的『環境變數』"""
    ticker: str
    dl_model_type: DLModelType = DLModelType.HYBRID
    rnn_type: RNNType = RNNType.LSTM
    # 除非重新尋找模型超參數，否則不可調整 lookahead
    lookahead: int = 10

@dataclass(frozen=True)
class EntryQualityCriteria:
    ''' 進場品質準則 '''
    ATR_LOOKBACK: int = 14            # 波動率參考週期
    PROFIT_TARGET_ATR: float = 3.0    # 獲利觸發倍數 (MFE)
    STOP_LOSS_ATR: float = 1.5        # 停損容忍倍數 (MAE)

@dataclass(frozen=True)
class MarketRiskCriteria:
    ''' 大盤防禦準則 (第三腦專用) '''
    ATR_LOOKBACK: int = 14
    # 未來 20 天跌幅超過 1.5 倍 ATR
    CRASH_THRESHOLD_ATR: float = 1.5


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

    # 轉折與極值特徵參數 (Reversal & Extremes)
    DONCHIAN_WINDOW: int = 20  # 區間高低點 (Donchian Channel) 的計算天數
    KD_RSV_WINDOW: int = 9     # KD 指標的 RSV 計算天數 (經典參數為 9)
    KD_SMOOTH: int = 3         # KD 指標的平滑天數 (經典參數為 3，代表 1/3 權重)


@dataclass(frozen=True)
class XGBHyperParams:
    OBJECTIVE: str = 'binary:logistic'  # 輸出 0~1 的機率
    EVAL_METRIC: str = 'auc'            # 使用 AUC 評估模型排序能力
    N_ESTIMATORS: int = 100
    RANDOM_STATE: int = 42              # 固定亂數種子，確保結果可重現
    N_JOBS: int = 1
    MAX_DEPTH: int = 3                  # 限制樹的深度，防止過度擬合 (Overfitting)
    MIN_CHILD_WEIGHT: int = 3
    LEARNING_RATE: float = 0.0992
    SUBSAMPLE: float = 0.6505           # 每次建樹只用 80% 的樣本 (增加泛化能力)
    COLSAMPLE_BYTREE: float = 0.7472    # 每次建樹只用 80% 的特徵
    GAMMA: float = 3.4100
    REG_ALPHA: float = 2.2545
    REG_LAMBDA: float = 0.8625

@dataclass(frozen=True)
class TrainConfig:
    N_SPLITS: int = 5
    N_SPLITS_MAX: int = 8
    N_SPLITS_MIN: int = 3

    ML_EARLY_STOP_ROUND = 25
    DL_EARLY_STOP_ROUND = 10

@dataclass(frozen=True)
class DLHyperParams:
    """CNN-LSTM 深度學習超參數"""
    INPUT_SIZE: int = 11
    TIME_STEPS: int = 20                # 滑動窗口大小 (回顧過去 ~ 天)
    CNN_OUT_CHANNELS: int = 32          # CNN 特徵提取後的維度
    RNN_HIDDEN: int = 16                # RNN 隱藏層神經元數量
    NUM_LAYERS: int = 1                 # LSTM/GRU 疊了幾層
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.004986
    EPOCHS: int = 50
    DROPOUT: float = 0.499119           # 隨機失活率
    SCHEDULER_PATIENCE: int = 3
    SCHEDULER_FACTOR: float = 0.5
    KERNEL_SIZE: int = 2                # 降採樣的倍率

@dataclass
class MarketLGBMConfig:
    """LightGBM 大盤防禦模型超參數配置 - Optuna 優化版本"""
    OBJECTIVE: str = 'binary'
    METRIC: str = 'auc'
    BOOSTING_TYPE: str = 'gbdt'

    # 核心結構參數
    MAX_DEPTH: int = 3               # 原本: 4
    NUM_LEAVES: int = 4              # 新增: 限制葉子數以防過擬合
    MIN_CHILD_SAMPLES: int = 16      # 新增: 確保每個節點有足夠樣本
    MIN_SPLIT_GAIN: float = 0.7      # 新增: 極高門檻，強迫模型只抓強訊號 (4.4533)

    # 學習與正則化
    LEARNING_RATE: float = 0.01      # 0.0029
    N_ESTIMATORS: int = 100          # 原本: 100 (註: 配合低學習率，實盤可視情況增加)
    SUBSAMPLE: float = 0.7428        # 原本: 0.8
    COLSAMPLE_BYTREE: float = 0.4491 # 原本: 0.8 (對應 feature_fraction)

    # 防震盪正則化
    REG_ALPHA: float = 2.0228        # L1 正則化
    REG_LAMBDA: float = 0.3432       # L2 正則化
    MAX_BIN: int = 255               # 特徵分桶數

    RANDOM_STATE: int = 42
    VERBOSE: int = -1

    @property
    def to_dict(self) -> Dict[str, Any]:
        """轉換為 LightGBM 吃的字典格式，並排除非原生參數"""
        return asdict(self)

@dataclass(frozen=True)
class MetaHyperParams:
    # 核心超參數 (用於 Tuning)
    C: float = 0.5
    PENALTY: str = "l2"
    CLASS_WEIGHT: str = "balanced"

    # 穩定性配置 (通常不需變動)
    RANDOM_STATE: int = 42

    # 演算法配置 (通常不需變動)
    SOLVER: str = "lbfgs"
    MAX_ITER: int = 100
