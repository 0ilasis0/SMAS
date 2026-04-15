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
class SubPortfolio:
    """
    獨立的投資組合包
    管理自己的持股清單與專屬資金 (如果有的話)。
    """
    name: str = "新增組合"
    use_shared_cash: bool = True  # True: 使用未分配流動資金, False: 使用專屬資金
    allocated_cash: float = 0.0   # 專屬資金餘額 (僅在 use_shared_cash=False 時有效)
    watch_tickers: List[str] = field(default_factory=list) # 此組合關注的股票
    positions: Dict[str, Position] = field(default_factory=dict) # 此組合的實際庫存

    def get_position(self, ticker: str) -> Position:
        """取得特定標的持倉，若無則回傳空持倉"""
        if ticker not in self.positions:
            self.positions[ticker] = Position()
        return self.positions[ticker]

    @property
    def total_market_value(self) -> float:
        """計算此組合包的總市值"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_cost_value(self) -> float:
        """計算此組合包的總成本"""
        return sum(pos.cost_value for pos in self.positions.values())


@dataclass
class Account:
    """
    實體帳戶/資金組合 (Portfolio)。
    管理全域現金與所有持倉。
    """
    total_cash: float = 0.0
    sub_portfolios: Dict[str, SubPortfolio] = field(default_factory=dict)
    # positions: Dict[str, Position] = field(default_factory=dict)

    @property
    def unallocated_cash(self) -> float:
        """
        計算活資金 (總資金可用餘額)。
        等於「系統總現金」減去「所有被劃撥出去的專屬資金」。
        """
        allocated_sum = sum(p.allocated_cash for p in self.sub_portfolios.values() if not p.use_shared_cash)
        return max(0.0, self.total_cash - allocated_sum)

    # def get_position(self, ticker: str) -> Position:
    #     """取得特定標的的持倉，若無則回傳空持倉"""
    #     if ticker not in self.positions:
    #         self.positions[ticker] = Position()
    #     return self.positions[ticker]

    # def update_price(self, ticker: str, current_price: float):
    #     """更新特定標的的最新報價 (供市值計算用)"""
    #     pos = self.get_position(ticker)
    #     pos.current_price = current_price

    @property
    def total_market_value(self) -> float:
        """計算所有組合包的總市值"""
        return sum(sp.total_market_value for sp in self.sub_portfolios.values())

    @property
    def total_cost_value(self) -> float:
        """計算所有組合包的總成本"""
        return sum(sp.total_cost_value for sp in self.sub_portfolios.values())

    @property
    def total_equity(self) -> float:
        """【核心指標】計算總資產淨值 (系統總現金 + 總市值)"""
        all_equity = self.total_cash + self.total_market_value
        if all_equity <= 0 and self.total_cash > 0:
            dbg.war(f"總權益異常: {all_equity}")
        return all_equity

    def get_sub_portfolio(self, sp_id: str) -> SubPortfolio:
        """取得特定組合包，若不存在則報錯或建立預設 (依實作而定)"""
        if sp_id not in self.sub_portfolios:
            # 安全防護：如果找不到，給他一個空的
            self.sub_portfolios[sp_id] = SubPortfolio(name=sp_id)
        return self.sub_portfolios[sp_id]