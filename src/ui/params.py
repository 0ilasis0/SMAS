from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestParams:
    MAX_DAYS = 240
    MIN_DAYS = 60
    STEP_DAYS = 10
