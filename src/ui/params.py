from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestParams:
    MAX_DAYS: int = 240
    MIN_DAYS: int = 60
    STEP_DAYS: int = 10


@dataclass(frozen=True)
class AcountLimit:
    MAX_MONEY: int = 100000000
    MIN_MONEY: int = 0
    STEP_MONEY: int = 10000
