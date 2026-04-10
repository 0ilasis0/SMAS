from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestParams:
    """回測 UI 控制項的預設範圍"""
    MAX_DAYS: int = 240
    MIN_DAYS: int = 60
    STEP_DAYS: int = 10
    DEFAULT_DAYS: int = 240


@dataclass(frozen=True)
class AccountLimit:
    """帳戶資金 UI 控制項的預設範圍"""
    MAX_MONEY: int = 10_000_000_000
    MIN_MONEY: int = 0
    STEP_MONEY: int = 10000
    DEFAULT_SINGLE: int = 300_000
    DEFAULT_GLOBAL: int = 2_000_000
