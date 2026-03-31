from enum import StrEnum


class TimeUnit(StrEnum):
    YEAR = 'y'
    MONTH = 'mo'
    DAY = 'd'

class YfInterval(StrEnum):
    DAILY = "1d"
    INTRADAY_5M = "2m"


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


class MacroTicker(StrEnum):
    """總經與大盤指數標的"""
    TWII = "^TWII"  # 台灣加權指數 (本地)
    SOX = "^SOX"    # 費城半導體指數 (海外，需處理時差)

    @classmethod
    def get_overseas_tickers(cls) -> list[str]:
        """回傳需要進行 T-1 時差處理的海外標的"""
        return [cls.SOX]
