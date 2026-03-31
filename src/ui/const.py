from enum import StrEnum


class EncodingConst(StrEnum):
    STD_FONT = "utf-8"


class Page(StrEnum):
    DASHBOARD = "dashboard"
    PORTFOLIO = "portfolio"


class PortfolioCol(StrEnum):
    GLOBAL_CASH = "global_cash"
    POSITIONS = "positions"
    SHARES = "shares"
    AVG_COST = "avg_cost"
    HISTORY = "history"
