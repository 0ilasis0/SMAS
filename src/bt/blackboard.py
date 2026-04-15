from dataclasses import dataclass, field
from typing import Any, Dict

from bt.account import Account
from bt.const import TradeDecision
from bt.params import LLMParams, TaxRate
from const import GlobalParams


@dataclass
class Blackboard:
    """
    黑板系統 (Blackboard)。
    行為樹中所有節點共享的記憶體中心，用來存放環境狀態、預測機率、帳戶資金與交易紀錄。
    """
    # 基本資訊
    is_backtest: bool = False
    ticker: str = ""
    current_date: str = ""
    current_price: float = 0.0

    # 預測勝率 (來自 QuantAIEngine)
    prob_xgb: float = GlobalParams.DEFAULT_ERROR
    prob_dl: float = GlobalParams.DEFAULT_ERROR
    prob_final: float = GlobalParams.DEFAULT_ERROR
    prob_market_safe: float = GlobalParams.DEFAULT_ERROR

    cooldown_timer: int = 0        # 停損冷卻期倒數天數

    # AI 分析與決策結果
    oracle: Any = None
    action_decision: str = TradeDecision.HOLD
    gemini_reasoning: str = ""     # Gemini 產出的分析報告
    sentiment_score: int = LLMParams.DEFAULT_SENTIMENT_SCORE
    sentiment_reason: str = ""     # 新聞情緒理由

    last_trade_shares: int = 0
    last_trade_profit: float = 0.0
    last_trade_price: float = 0.0

    # 帳戶與持股狀態
    account: Account = None     # 可用資金
    position: int = 0           # 持有股數
    avg_cost: float = 0.0       # 個股持倉平均成本
    highest_price: float = 0.0  # 移動停損專用的最高價記憶

    # 交易執行用的現實環境變數
    executable_price: float = 0.0
    daily_volume: float = 0.0
    bias_20: float = 0.0
    return_5d: float = 0.0

    # 波段交易記憶
    entry_count: int = 0                  # 紀錄這個波段總共「買進/加碼」了幾次
    is_partial_profit_taken: bool = False # 紀錄是否已經觸發過「部分停利」

    # 動態擴充區 (給特殊的節點放臨時變數)
    context: Dict[str, Any] = field(default_factory=dict)

    # 快取
    cached_return_rate: float | None = None

    def set(self, key: str, value: Any):
        self.context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def update_price(self, current_price: float, high_price: float, executable_price: float, daily_volume: float):
        self.current_price = current_price
        self.executable_price = executable_price
        self.daily_volume = daily_volume
        self.cached_return_rate = None

        # 移動停損最高價追蹤 (使用高點，更精準)
        if self.position > 0:
            self.highest_price = max(self.highest_price, high_price)
        else:
            self.highest_price = 0.0

    def clear_trade_memory(self):
        """當部位全數出清時，重置所有波段記憶與成本"""
        self.position = 0
        self.avg_cost = 0.0
        self.highest_price = 0.0
        self.entry_count = 0
        self.is_partial_profit_taken = False
        self.cached_return_rate = None

    @property
    def has_position(self) -> bool:
        """判斷目前是否持有該檔股票"""
        return self.position > 0

    @property
    def holding_ratio(self) -> float:
        """
        計算當前此檔股票的曝險比例 (持倉市值 / 總資產)
        此屬性供「動態風控節點」評估是否過度集中。
        """
        if self.account is None or self.position <= 0: return 0.0

        # 取得當前這檔股票的市值
        position_value = self.position * self.current_price
        # Account 的總資產！
        total_equity = self.account.total_equity

        return position_value / total_equity

    @property
    def estimated_return_rate(self) -> float:
        """計算真實報酬率 (包含緩存機制)"""
        if self.cached_return_rate is not None:
            return self.cached_return_rate

        if self.position <= 0 or self.avg_cost <= 0:
            self.cached_return_rate = 0.0
            return 0.0

        raw_revenue = self.position * self.current_price
        fee = max(TaxRate.MIN_FEE, raw_revenue * TaxRate.FEE_RATE)
        tax = raw_revenue * TaxRate.TAX_RATE

        actual_revenue = raw_revenue - fee - tax
        total_cost = self.position * self.avg_cost

        profit = actual_revenue - total_cost

        self.cached_return_rate = profit / total_cost
        return self.cached_return_rate

    @property
    def cash(self) -> float:
        """提供一個捷徑屬性，方便原本的 Action 節點讀取"""
        if self.account is None: return 0.0
        return self.account.total_cash

    @cash.setter
    def cash(self, value: float):
        """提供一個捷徑屬性，方便原本的 Action 節點扣款"""
        if self.account is not None:
            self.account.total_cash = value
