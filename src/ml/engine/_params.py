from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """市場極端風險斷路器閥值設定"""
    # 費半單日暴跌閥值
    SOX_CRASH_THRESHOLD: float = -0.04
    # VIX 單日飆升閥值
    VIX_PANIC_THRESHOLD: float = 0.25
    # EMA 平滑係數 (用於控制機率波動)
    SMOOTH_SPAN: int = 3
