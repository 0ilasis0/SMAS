import yfinance as yf

from bt.const import TradeDecision


def is_valid_ticker(ticker: str) -> bool:
    """輕量級驗證：檢查 Yahoo Finance 是否有該標的之資料"""
    try:
        # 使用 history 抓取 1 天資料，速度最快且不會印出不必要的 log
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1d")
        return not df.empty
    except Exception:
        return False



class UIActionMapper:
    """UI 動作顯示對映表管理器 (中英雙向轉換)"""
    _FORWARD = {
        TradeDecision.BUY.value: "🟢 買進",
        TradeDecision.SELL.value: "🔴 賣出"
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


