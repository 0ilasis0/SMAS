from dataclasses import asdict, dataclass
from typing import Any, Dict

from ml.const import DLModelType, RNNType


@dataclass
class SessionConfig:
    """存放當前任務的『環境變數』"""
    ticker: str
    dl_model_type: DLModelType = DLModelType.HYBRID
    rnn_type: RNNType = RNNType.LSTM
    lookahead: int = 20


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


@dataclass(frozen=True)
class XGBHyperParams:
    objective: str = 'binary:logistic'  # 輸出 0~1 的機率
    eval_metric: str = 'auc'            # 使用 AUC 評估模型排序能力
    max_depth: int = 4                  # 限制樹的深度，防止過度擬合 (Overfitting)
    learning_rate: float = 0.05
    n_estimators: int = 100
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
    INPUT_SIZE: int = 8
    TIME_STEPS: int = 20                # 滑動窗口大小 (回顧過去 ~ 天)
    CNN_OUT_CHANNELS: int = 16          # CNN 特徵提取後的維度
    LSTM_HIDDEN: int = 32               # LSTM 隱藏層神經元數量
    NUM_LAYERS: int = 1
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.002
    EPOCHS: int = 50
    DROPOUT: float = 0.2
    SCHEDULER_PATIENCE: int = 3
    SCHEDULER_FACTOR: float = 0.5
    KERNEL_SIZE: int = 2

@dataclass
class MarketLGBMConfig:
    """LightGBM 大盤防禦模型超參數配置"""
    objective: str = 'binary'
    metric: str = 'auc'
    boosting_type: str = 'gbdt'
    max_depth: int = 4              # 限制深度防止過擬合
    learning_rate: float = 0.05
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    verbose: int = -1
    early_stopping_rounds: int = 30 # 將 callback 參數也納入管理

    def to_dict(self) -> Dict[str, Any]:
        """轉換為 LightGBM 吃的字典格式，並排除非原生參數"""
        data = asdict(self)
        # 移除不屬於 LGBMClassifier 構造函數的參數
        data.pop('early_stopping_rounds', None)
        return data

@dataclass(frozen=True)
class MetaHyperParams:
    # 核心超參數 (用於 Tuning)
    C: float = 0.3
    PENALTY: str = 'l2'
    CLASS_WEIGHT: str = 'balanced'

    # 穩定性配置 (通常不需變動)
    RANDOM_STATE: int = 42

    # 演算法配置 (通常不需變動)
    SOLVER: str = 'lbfgs'
    MAX_ITER: int = 100
