from enum import StrEnum


# ==========================================
# 黑板記憶體 (Blackboard) 變數定義
# ==========================================
class BlackboardKey(StrEnum):
    """
    這裡專門用來定義透過字典 (bb.get()) 存取的鍵值。
    """
    ORACLE = "oracle"
    COOLDOWN_TIMER = "cooldown_timer"

# ==========================================
# 最終交易決策定義
# ==========================================
class TradeDecision(StrEnum):
    """
    系統最終產出的交易決策狀態，用於 UI 顯示、下單 API、回測結算。
    """
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

# ==========================================
# 行為樹 (Behavior Tree) 節點名稱定義
# ==========================================
class BTAction(StrEnum):
    """
    行為樹的「執行節點 (Action Node)」
    """
    EXECUTE_BUY = "execute_buy"
    EXECUTE_SELL = "execute_sell"
    EXECUTE_HOLD = "execute_hold"
    GENERATE_REPORT = "generate_gemini_report"

class BTCondition(StrEnum):
    """
    行為樹的「條件節點 (Condition Node)」名稱
    """
    NOT_OVERHEATED = "check_not_overheated"
    COOLDOWN = "check_cooldown"
    TREND_FILTER = "check_trend_filter"
    SENTIMENT_FILTER = "check_sentiment_filter"
    SELL_SENTIMENT_FILTER = "check_sell_sentiment_filter"
    GAP_LIMIT = "check_gap_limit"
    NOT_PARTIAL_TAKEN = "check_not_partial_taken"
    ENTRY_COUNT_LIMIT = "check_entry_count_limit"
    HAS_POSITION = "check_has_position"
    BUY_SIGNAL = "check_buy_signal"
    SELL_SIGNAL = "check_sell_signal"
    STOP_LOSS = "check_stop_loss"
    TAKE_PROFIT = "check_take_profit"
    TRAILING_STOP = "check_trailing_stop"