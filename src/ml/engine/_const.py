from dataclasses import dataclass

from data.const import TimeUnit
from data.params import DataLimit


@dataclass(frozen=True)
class DataConst:
    RETURNS_ABS: int = 0.45

    # 數據自動修復的專屬配置區
    HEAL_PERIOD: int = DataLimit.DAILY_DEFAULT_YEAR // 2
    HEAL_UNIT: TimeUnit = TimeUnit.YEAR

    # 手動平滑演算法的合理變動倍數邊界
    HEAL_MIN_RATIO: float = 0.05  # 超過這個縮水幅度，視為數值損毀而非除權息
    HEAL_MAX_RATIO: float = 20.0  # 超過這個暴漲幅度，視為數值損毀
