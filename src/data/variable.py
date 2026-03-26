from enum import StrEnum

from base import FrozenMeta


class DataLimit(metaclass=FrozenMeta):
    DAILY_MAX_YEAR = 10
    DAILY_MAX_MONTH = 120
    INTRADAY_MAX_DAY = 60

class TimeUnit(StrEnum):
    YEAR = 'y'
    MONTH = 'mo'
    DAY = 'd'

class YfInterval(StrEnum):
    DAILY = "1d"
    INTRADAY_5M = "5m"

class StockCol(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TICKER = "ticker"
    DATE = "date"

    @classmethod
    def get_ohlcv(cls):
        """回傳所有要餵給 AI 學習的特徵欄位清單"""
        return [cls.OPEN, cls.HIGH, cls.LOW, cls.CLOSE, cls.VOLUME]
