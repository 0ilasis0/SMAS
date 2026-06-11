from base import FrozenMeta


class DataLimit(metaclass=FrozenMeta):
    MARKET_DEFAULT_YEAR: int = 15
    DAILY_DEFAULT_YEAR: int = 20
    DAILY_MAX_YEAR: int = 40
    DAILY_MAX_MONTH: int = 480

    INTRADAY_DEFAULT_DAY: int = 90
    INTRADAY_MAX_DAY: int = 90
