from dataclasses import dataclass


@dataclass(frozen=True)
class TaxRate:
    ''' 台灣股市基礎費率設定 (可依券商折讓自行調整) '''
    FEE_RATE: float = 0.001425  # 券商手續費率 (買賣皆收)
    TAX_RATE: float = 0.003     # 證券交易稅率 (僅賣出收取)
    MIN_FEE: float = 20.0       # 手續費低消

@dataclass(frozen=True)
class ConsiderConfig:
    # 限制買入與賣出的最大勝率
    MAX_SELL_THRESHOLD: float = 0.5
    MAX_BUY_THRESHOLD: float = 0.9

@dataclass(frozen=True)
class LLMParams:
    DEFAULT_SENTIMENT_SCORE = 5
