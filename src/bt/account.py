from dataclasses import dataclass, field
from typing import Dict, List

from debug import dbg


@dataclass
class Position:
    """單一檔股票的持倉資訊"""
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0 # 供系統隨時更新最新報價
    history: List[dict] = field(default_factory=list) # 儲存交易紀錄

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_value(self) -> float:
        return self.shares * self.avg_cost

@dataclass
class Account:
    """
    實體帳戶/資金組合 (Portfolio)。
    管理全域現金與所有持倉。
    """
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)

    def get_position(self, ticker: str) -> Position:
        """取得特定標的的持倉，若無則回傳空持倉"""
        if ticker not in self.positions:
            self.positions[ticker] = Position()
        return self.positions[ticker]

    def update_price(self, ticker: str, current_price: float):
        """更新特定標的的最新報價 (供市值計算用)"""
        pos = self.get_position(ticker)
        pos.current_price = current_price

    @property
    def total_market_value(self) -> float:
        """計算所有股票的總市值"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_cost_value(self) -> float:
        """計算所有股票的總成本"""
        return sum(pos.cost_value for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """【核心指標】計算總資產淨值 (現金 + 總市值)"""
        all_equity = self.cash + self.total_market_value
        if all_equity <= 0:
            dbg.error(f"total_equity:{self.cash + self.total_market_value} <= 0")
            return 0.0
        return all_equity