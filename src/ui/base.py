import streamlit as st
import yfinance as yf

from bt.const import TradeDecision


def is_valid_ticker(ticker: str) -> bool:
    """輕量級驗證：檢查 Yahoo Finance 是否有該標的之資料"""
    try:
        ticker_obj = yf.Ticker(ticker)
        # 如果這檔股票不存在，讀取 fast_info['regularMarketPrice'] 會觸發 Exception
        price = ticker_obj.fast_info['regularMarketPrice']
        return price is not None and price > 0
    except Exception:
        try:
            df = ticker_obj.history(period="1d")
            return not df.empty
        except Exception:
            return False

def get_smart_tw_ticker(raw_ticker: str) -> str | None:
    """
    智慧判斷台灣股票代號 (上市/上櫃)。
    回傳正確的 Yahoo Finance 代號，若皆無效則回傳 None。
    """
    clean_ticker = raw_ticker.strip().upper()

    # 1. 如果使用者已經手動加上後綴 (如 AAPL, 2330.TW, 3105.TWO)，直接驗證
    if "." in clean_ticker or not clean_ticker.isdigit():
        if not clean_ticker.endswith(".TW") and not clean_ticker.endswith(".TWO") and clean_ticker.isdigit():
            clean_ticker += ".TW" # 預防使用者打 2330.

        if is_valid_ticker(clean_ticker):
            return clean_ticker
        return None

    # 先猜測是上市股票 (.TW)
    guess_tw = f"{clean_ticker}.TW"
    if is_valid_ticker(guess_tw):
        return guess_tw

    # 如果上市找不到，自動猜測是上櫃股票 (.TWO)
    guess_two = f"{clean_ticker}.TWO"
    if is_valid_ticker(guess_two):
        return guess_two

    return None

class UIActionMapper:
    """UI 動作顯示對映表管理器 (中英雙向轉換)"""
    _FORWARD = {
        TradeDecision.BUY.value: "🟢 買進",
        TradeDecision.SELL.value: "🔴 賣出",
    }
    _REVERSE = {v: k for k, v in _FORWARD.items()}

    @classmethod
    def get_map(cls) -> dict:
        """取得底層到 UI 的轉換字典 (供 Pandas map 使用)"""
        return cls._FORWARD

    @classmethod
    def get_options(cls) -> list[str]:
        """取得所有 UI 選項清單 (供 Streamlit Selectbox/Radio 使用)"""
        return list(cls._FORWARD.values())

    @classmethod
    def to_core(cls, ui_action: str, default: str = TradeDecision.BUY.value) -> str:
        """將 UI 中文選項轉換回底層 Enum 原始值 (小寫英文)"""
        return cls._REVERSE.get(ui_action, default)

    @classmethod
    def is_buy(cls, ui_action: str) -> bool:
        """判斷當前 UI 選擇是否為買進"""
        return ui_action == cls._FORWARD[TradeDecision.BUY.value]


