from enum import StrEnum


class BtVar(StrEnum):
    BASE_NODE = "base_node"
    GENERATE_GEMINI_REPORT = "generate_gemini_report"


class ConditionCol(StrEnum):
    CHECK_HAS_POSITION = "check_has_position"
    CHECK_BUY_SIGNAL = "check_buy_signal"
    CHECK_SELL_SIGNAL = "check_sell_signal"
    CHECK_STOP_LOSS = "check_stop_loss"
    CHECK_TAKE_PROFIT = "check_take_profit"
    CHECK_TRAILING_STOP = "check_trailing_stop"

class ExecuteCol(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DecisionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
