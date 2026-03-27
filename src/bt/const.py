from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class BtVar:
    BASE_NODE = "base_node"
    GENERATE_GEMINI_REPORT = "generate_gemini_report"
    TRADE_UNIT = 1000

class ExecuteCol(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class ConditionCol(StrEnum):
    CHECK_NOT_PARTIAL_TAKEN = "check_not_partial_taken"
    CHECK_ENTRY_COUNT_LIMIT = "check_entry_count_limit"
    CHECK_HAS_POSITION = "check_has_position"
    CHECK_BUY_SIGNAL = "check_buy_signal"
    CHECK_SELL_SIGNAL = "check_sell_signal"
    CHECK_STOP_LOSS = "check_stop_loss"
    CHECK_TAKE_PROFIT = "check_take_profit"
    CHECK_TRAILING_STOP = "check_trailing_stop"

class StrategyCol(StrEnum):
    DEFENSIVE = "defensive"

class DecisionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
