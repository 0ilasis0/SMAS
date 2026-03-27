from enum import StrEnum


class TimeUnit(StrEnum):
    YEAR = 'y'
    MONTH = 'mo'
    DAY = 'd'

class YfInterval(StrEnum):
    DAILY = "1d"
    INTRADAY_5M = "3m"

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
