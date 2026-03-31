from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class BtVar:
    BASE_NODE = "base_node"
    GENERATE_GEMINI_REPORT = "generate_gemini_report"
    TRADE_UNIT = 1000
    COOLDOWN_TIMER = "cooldown_timer"
    DEFAULT_LLM_SCORE = 5


class LLMCol(StrEnum):
    SENTIMENT_SCORE = "sentiment_score"
    SENTIMENT_REASON = "sentiment_reason"

class ExecuteCol(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class ConditionCol(StrEnum):
    CHECK_NOT_OVERHEATED = "check_not_overheated_node"
    CHECK_COOLDOWN = "check_cooldown"
    CHECK_TREND_FILTER = "check_trend_filter"
    CHECK_SENTIMENT_FILTER = "check_sentiment_filter"
    CHECK_SELL_SENTIMENT_FILTER = "check_sell_sentiment_filter"
    CHECK_GAP_LIMIT = "check_gap_limit"
    CHECK_NOT_PARTIAL_TAKEN = "check_not_partial_taken"
    CHECK_ENTRY_COUNT_LIMIT = "check_entry_count_limit"
    CHECK_HAS_POSITION = "check_has_position"
    CHECK_BUY_SIGNAL = "check_buy_signal"
    CHECK_SELL_SIGNAL = "check_sell_signal"
    CHECK_STOP_LOSS = "check_stop_loss"
    CHECK_TAKE_PROFIT = "check_take_profit"
    CHECK_TRAILING_STOP = "check_trailing_stop"

class DecisionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
