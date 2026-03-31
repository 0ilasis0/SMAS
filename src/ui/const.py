from enum import StrEnum


class EncodingConst(StrEnum):
    UTF8 = "utf-8"


class Page(StrEnum):
    DASHBOARD = "dashboard"
    PORTFOLIO = "portfolio"


class PortfolioCol(StrEnum):
    GLOBAL_CASH = "global_cash"
    POSITIONS = "positions"
    SHARES = "shares"
    AVG_COST = "avg_cost"
    HISTORY = "history"
