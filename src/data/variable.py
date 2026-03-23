from enum import StrEnum

from base import FrozenMeta


class DataLimit(metaclass=FrozenMeta):
    DAILY_MAX_YEAR = 5
    DAILY_MAX_MONTH = 60
    INTRADAY_MAX_DAY = 60

class TimeUnit(StrEnum):
    YEAR = 'y'
    MONTH = 'mo'
    DAY = 'd'

class YfInterval(StrEnum):
    DAILY = "1d"
    INTRADAY_5M = "5m"

class StockCol(StrEnum):
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    VOLUME = "Volume"
    TICKER = "ticker"
    DATE = "date"

