import json
import os

import streamlit as st

from bt.account import Account, AccountCol, Position, SubPortfolio
from bt.const import TradeDecision
from path import PathConfig
from ui.const import EncodingConst, PortfolioCol


def load_portfolio() -> Account:
    """從 JSON 讀取資料，並實例化為支援多組合包的 Account 物件"""
    account = Account()

    if not os.path.exists(PathConfig.PORTFOLIO):
        return account

    try:
        with open(PathConfig.PORTFOLIO, "r", encoding=EncodingConst.UTF8.value) as f:
            data = json.load(f)

        # 讀取檔案
        account.total_cash = data.get(AccountCol.TOTAL_CASH.value, 0.0)
        sp_data = data.get(AccountCol.SUB_PORTFOLIOS.value, {})

        for sp_id, sp_dict in sp_data.items():
            sp = SubPortfolio(
                name=sp_dict.get("name", sp_id),
                use_shared_cash=sp_dict.get("use_shared_cash", True),
                allocated_cash=sp_dict.get("allocated_cash", 0.0),
                watch_tickers=sp_dict.get("watch_tickers", [])
            )

            # 解析持倉
            positions_data = sp_dict.get("positions", {})
            for ticker, pos_dict in positions_data.items():
                sp.positions[ticker] = Position(
                    shares=pos_dict.get(PortfolioCol.SHARES.value, 0),
                    avg_cost=pos_dict.get(PortfolioCol.AVG_COST.value, 0.0),
                    current_price=pos_dict.get(PortfolioCol.AVG_COST.value, 0.0), # 預設先用成本價
                    history=pos_dict.get(PortfolioCol.HISTORY.value, [])
                )
            account.sub_portfolios[sp_id] = sp

    except Exception as e:
        st.error(f"讀取資金檔發生錯誤，將使用空帳戶: {e}")

    return account

def _migrate_legacy_to_v2(legacy_data: dict) -> Account:
    """將舊版 JSON 自動升級為 V2 (裝入預設組合包)"""
    account = Account()
    # 舊版的 GLOBAL_CASH 變成新版的 total_cash
    account.total_cash = legacy_data.get(PortfolioCol.GLOBAL_CASH.value, 0.0)

    # 建立一個「預設組合包」來收留舊資產
    legacy_sp = SubPortfolio(
        name="預設組合",
        use_shared_cash=True, # 預設使用共用總資金
        allocated_cash=0.0
    )

    legacy_sp.watch_tickers = []

    # 轉移庫存
    positions_data = legacy_data.get(PortfolioCol.POSITIONS.value, {})
    for ticker, pos_dict in positions_data.items():
        legacy_sp.positions[ticker] = Position(
            shares=pos_dict.get(PortfolioCol.SHARES.value, 0),
            avg_cost=pos_dict.get(PortfolioCol.AVG_COST.value, 0.0),
            history=pos_dict.get(PortfolioCol.HISTORY.value, [])
        )

    account.sub_portfolios["Legacy_Portfolio_01"] = legacy_sp
    return account

def save_portfolio(account: Account):
    """將 Account 物件序列化回 JSON 儲存 (v2 格式)"""
    try:
        os.makedirs(os.path.dirname(PathConfig.PORTFOLIO), exist_ok=True)

        data_to_save = {
            AccountCol.VERSION.value: "2.0",
            AccountCol.TOTAL_CASH.value: account.total_cash,
            AccountCol.SUB_PORTFOLIOS.value: {}
        }

        for sp_id, sp in account.sub_portfolios.items():
            sp_dict = {
                "name": sp.name,
                "use_shared_cash": sp.use_shared_cash,
                "allocated_cash": sp.allocated_cash,
                "watch_tickers": sp.watch_tickers,
                "positions": {}
            }

            for ticker, pos in sp.positions.items():
                sp_dict["positions"][ticker] = {
                    PortfolioCol.SHARES.value: pos.shares,
                    PortfolioCol.AVG_COST.value: pos.avg_cost,
                    PortfolioCol.HISTORY.value: pos.history
                }

            data_to_save[AccountCol.SUB_PORTFOLIOS.value][sp_id] = sp_dict

        with open(PathConfig.PORTFOLIO, "w", encoding=EncodingConst.UTF8.value) as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    except Exception as e:
        st.error(f"❌ 資金檔存檔失敗: {e}")

def recalculate_position(history: list) -> tuple[int, float]:
    """動態帳務重算引擎 (嚴格平均成本法)"""
    from ui.const import HistoryKey
    current_shares = 0
    current_total_cost = 0.0
    STANDARD_BUY = TradeDecision.BUY.value
    STANDARD_SELL = TradeDecision.SELL.value

    for r in history:
        action = str(r.get(HistoryKey.ACTION.value, STANDARD_BUY)).lower()
        shares = int(r.get(HistoryKey.SHARES.value, 0))
        total_settlement = float(r.get(HistoryKey.TOTAL.value, 0.0))

        if action == STANDARD_BUY:
            current_shares += shares
            current_total_cost += total_settlement
        elif action == STANDARD_SELL:
            if current_shares > 0:
                avg_cost = current_total_cost / current_shares
                current_shares -= shares
                if current_shares <= 0:
                    current_shares = 0
                    current_total_cost = 0.0
                else:
                    current_total_cost = current_shares * avg_cost

    final_avg_cost = current_total_cost / current_shares if current_shares > 0 else 0.0
    return current_shares, final_avg_cost

def get_active_buys(history: list) -> list:
    """找出構成「當前庫存均價」的所有有效買進批次"""
    from ui.const import HistoryKey
    temp_shares = 0
    last_zero_idx = -1
    STANDARD_BUY = TradeDecision.BUY.value

    for i, r in enumerate(history):
        if str(r.get(HistoryKey.ACTION.value)).lower() == STANDARD_BUY:
            temp_shares += int(r.get(HistoryKey.SHARES.value, 0))
        else:
            temp_shares -= int(r.get(HistoryKey.SHARES.value, 0))
            if temp_shares <= 0:
                temp_shares = 0
                last_zero_idx = i

    active_buys = []
    for i in range(last_zero_idx + 1, len(history)):
        r = history[i]
        if str(r.get(HistoryKey.ACTION.value)).lower() == STANDARD_BUY:
            active_buys.append(r)

    return active_buys