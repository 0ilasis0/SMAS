from dataclasses import dataclass, field
from typing import Any, Dict

from bt.variable import DecisionAction


@dataclass
class Blackboard:
    """
    黑板系統 (Blackboard)。
    行為樹中所有節點共享的記憶體中心，用來存放環境狀態、預測機率、帳戶資金與交易紀錄。
    """
    # 基本資訊
    ticker: str = ""
    current_date: str = ""
    current_price: float = 0.0

    # 預測勝率 (來自 QuantAIEngine)
    prob_xgb: float = 0.5
    prob_dl: float = 0.5
    prob_final: float = 0.5

    last_trade_shares: int = 0
    last_trade_profit: float = 0.0
    last_trade_price: float = 0.0

    # 帳戶與持股狀態
    cash: float = 0                         # 可用資金
    position: list[str, int] = ["", 0]      # 持有個股與股數
    avg_cost: list[str, float] = ["", 0.0]  # 持有個股與持倉平均成本

    # AI 分析與決策結果
    action_decision: str = DecisionAction.HOLD
    gemini_reasoning: str = ""    # Gemini 產出的分析報告

    # 動態擴充區 (給特殊的節點放臨時變數)
    context: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any):
        self.context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    @property
    def has_position(self) -> bool:
        """判斷目前是否持有該檔股票"""
        return self.position > 0
