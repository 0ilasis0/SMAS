from dataclasses import dataclass


@dataclass(frozen=True)
class TaxRate:
    ''' 台灣股市基礎費率設定 (可依券商折讓自行調整) '''
    FEE_RATE: float = 0.001425  # 券商手續費率 (買賣皆收)
    TAX_RATE: float = 0.003     # 證券交易稅率 (僅賣出收取)
    MIN_FEE: float = 20.0       # 手續費低消

@dataclass(frozen=True)
class ConsiderVar:
    # 勝率
    BUY_THRESHOLD: float = 0.6
    SELL_THRESHOLD: float = 0.4

    # -0.1 代表跌 10% 就停損
    LOSS_TOLERANCE: float = -0.1
    PROFIT_TARGET: float = 0.2
    DRAWDOWN_TOLERANCE: float = -0.2

    # 動用可用資金的比例 (0.0 ~ 1.0)，1.0 為 All-in
    CAPITAL_RATIO: float = 0.5
    POSITION_RATIO:  float = 1.0

    # 摩擦成本
    MAX_FRICTION_COST_RATIO: float = 0.01
