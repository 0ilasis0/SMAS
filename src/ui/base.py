import yfinance as yf


def is_valid_ticker(ticker: str) -> bool:
    """輕量級驗證：檢查 Yahoo Finance 是否有該標的之資料"""
    try:
        # 使用 history 抓取 1 天資料，速度最快且不會印出不必要的 log
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1d")
        return not df.empty
    except Exception:
        return False
