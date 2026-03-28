from dataclasses import dataclass


@dataclass(frozen=True)
class TaxRate:
    ''' 台灣股市基礎費率設定 (可依券商折讓自行調整) '''
    FEE_RATE: float = 0.001425  # 券商手續費率 (買賣皆收)
    TAX_RATE: float = 0.003     # 證券交易稅率 (僅賣出收取)
    MIN_FEE: float = 20.0       # 手續費低消

@dataclass(frozen=True)
class ConsiderConfig:
    # 勝率
    BUY_THRESHOLD: float = 0.6
    SELL_THRESHOLD: float = 0.4

@dataclass
class StrategyConfig:
    """
    量化交易策略參數配置中心。
    將所有閥值集中管理，方便未來進行網格搜索 (Grid Search) 最佳化。
    """
    # ================= 防守與風險控管參數 =================
    stop_loss_tolerance: float = -0.05       # 強制停損容忍度 (-5%)
    trailing_stop_drawdown: float = -0.08    # 移動停損回落容忍度 (-8%)
    stop_loss_sell_ratio: float = 1.0        # 觸發停損時的賣出比例 (100% 全面撤退)

    sell_signal_threshold: float = 0.40      # AI 勝率低迷預警門檻 (<40%)
    warning_sell_ratio: float = 0.5          # AI 預警時的戰術減碼比例 (賣出 50%)

    take_profit_target: float = 0.30         # 極端停利目標 (+30%)
    take_profit_sell_ratio: float = 0.5      # 觸發極端停利時的減碼比例 (先入袋 50%)

    # ================= 進攻與資金控管參數 =================
    max_entries: int = 3                     # 單一波段最大加碼次數
    max_gap_ratio: int = 0.03

    strong_buy_threshold: float = 0.75       # 強烈買進訊號門檻 (>= 75%)
    strong_buy_capital_ratio: float = 1.0    # 強烈買進時動用的資金比例 (100%)

    conservative_buy_threshold: float = 0.60 # 保守買進訊號門檻 (>= 60%)
    conservative_buy_capital_ratio: float = 0.5 # 保守買進時動用的資金比例 (50%)