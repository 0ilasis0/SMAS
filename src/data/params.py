from base import FrozenMeta


class DataLimit(metaclass=FrozenMeta):
    DAILY_MAX_YEAR = 40
    DAILY_MAX_MONTH = 480
    INTRADAY_MAX_DAY = 90
