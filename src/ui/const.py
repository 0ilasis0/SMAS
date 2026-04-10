from enum import StrEnum


class EncodingConst(StrEnum):
    UTF8 = "utf-8"


class Page(StrEnum):
    """側邊欄導覽的頁面名稱"""
    DASHBOARD = "dashboard"
    PORTFOLIO = "portfolio"

class HistoryKey(StrEnum):
    """投資組合中「單筆交易紀錄」的字典鍵值"""
    DATE = "date"
    ACTION = "action"
    PRICE = "price"
    SHARES = "shares"
    FEE = "fee"
    TAX = "tax"
    TOTAL = "total"

class SessionKey(StrEnum):
    """統一管理 UI 狀態 (st.session_state) 的鍵值"""
    CURRENT_PAGE = "current_page"
    API_KEYS = "api_keys"
    CURRENT_TICKER = "current_ticker"
    PORTFOLIO = "portfolio"
    IS_TRAINING = "is_training"
    IS_GLOBAL_TRAINING = "is_global_training"
    WATCH_LIST = "watch_list"
    USER_SETTINGS = "user_settings"
    CTRL_LIVE = "ctrl_live"
    CTRL_BT = "ctrl_bt"
    UI_PERSONA = "ui_persona"
    UI_MODE = "ui_mode"
    LAST_RESULT = "last_result"


class APIKey(StrEnum):
    """ Controller 回傳給 UI 的標準化 JSON Key (API Contract) """
    STATUS = "status"
    MESSAGE = "message"
    MODE = "mode"
    PERSONA = "persona"

    DECISION = "decision"
    ACTION = "action"
    TRADE_SHARES = "trade_shares"
    TRADE_PRICE = "trade_price"

    ACCOUNT = "account_after_trade"
    CASH_LEFT = "cash_left"
    POSITION_LEFT = "position_left"
    TOTAL_EQUITY = "total_equity"

    AI_SIGNALS = "ai_signals"
    SENTIMENT = "sentiment"
    REPORT = "report"


class PortfolioCol(StrEnum):
    """投資組合資料庫/字典的鍵值"""
    GLOBAL_CASH = "global_cash"
    POSITIONS = "positions"
    SHARES = "shares"
    AVG_COST = "avg_cost"
    HISTORY = "history"


class UIFormat(StrEnum):
    """UI 常用的字串格式"""
    DATE_FORMAT = "%Y-%m-%d"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    CURRENCY_FORMAT = "{:,.0f}"  # 例如 1,000,000
    PERCENT_FORMAT = "{:.2%}"    # 例如 15.25%
