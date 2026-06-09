import os
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from const import GlobalCol, StatusCol
from debug import dbg
from path import PathConfig


class HybridEventFetcher:
    """
    事件抓取備援引擎 (FinMind 優先 -> yfinance 備援)。
    內建斷路器機制，若 FinMind 連續失敗，當日將永久切換至 yfinance。
    """
    # 紀錄 FinMind 是否已經掛掉
    _finmind_circuit_breaker_tripped = False

    def __init__(self):
        load_dotenv(PathConfig.ENV_FILE)
        self.finmind_token = os.getenv(GlobalCol.FINMIND_API_KEYS)

        self.finmind_url = "https://api.finmindtrade.com/api/v4/data"

    def fetch_upcoming_events(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """同時抓取除權息與法說會，回傳 (df_div, df_earn)"""

        # 如果斷路器已經觸發，直接跳過 FinMind
        if not self._finmind_circuit_breaker_tripped:
            df_div, df_earn = self._fetch_from_finmind(ticker)
            if not df_div.empty or not df_earn.empty:
                return df_div, df_earn

            # 如果 FinMind 回傳空值或報錯，觸發斷路器
            dbg.war("⚠️ FinMind 抓取失敗，觸發斷路器！今日後續更新將全面改用 yfinance。")
            HybridEventFetcher._finmind_circuit_breaker_tripped = True

        # 使用備援機制
        return self._fetch_from_yfinance(ticker)

    def _fetch_from_finmind(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        dbg.log(f"[{ticker}] 🌐 嘗試使用 [FinMind API] 抓取事件資料...")
        stock_id = ticker.replace(".TW", "").replace(".TWO", "")
        start_date = datetime.now().strftime("%Y-%m-%d")

        df_div = pd.DataFrame()
        df_earn = pd.DataFrame()

        try:
            params = {
                "dataset": "TaiwanStockDividend",
                "data_id": stock_id,
                "start_date": start_date,
                "token": self.finmind_token
            }
            resp = requests.get(self.finmind_url, params=params, timeout=10)
            data = resp.json()

            if data.get("msg") == StatusCol.SUCCESS and data.get("data"):
                df_raw = pd.DataFrame(data["data"])
                # 篩選未來的除息日
                upcoming = df_raw[df_raw["CashExDividendTradingDate"] >= start_date].copy()
                if not upcoming.empty:
                    df_div = pd.DataFrame({
                        'ticker': ticker,
                        'ex_date': upcoming["CashExDividendTradingDate"],
                        'cash_dividend': upcoming["CashDividend"]
                    })
                dbg.log(f"[{ticker}] ✅ [FinMind API] 抓取除權息成功！")
            return df_div, df_earn
        except Exception as e:
            dbg.error(f"[{ticker}] ❌ [FinMind API] 發生異常: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _fetch_from_yfinance(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        dbg.log(f"[{ticker}] 🛡️ 啟用備援機制，使用 [yfinance] 抓取事件資料...")
        df_div = pd.DataFrame()
        df_earn = pd.DataFrame()

        try:
            stock = yf.Ticker(ticker)
            today = datetime.now().strftime("%Y-%m-%d")

            # 抓除權息
            divs = stock.dividends
            if not divs.empty:
                df = divs.reset_index()
                df.columns = ['ex_date', 'cash_dividend']
                df['ex_date'] = df['ex_date'].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
                upcoming_div = df[df['ex_date'] >= today].copy()

                if not upcoming_div.empty:
                    upcoming_div['ticker'] = ticker
                    df_div = upcoming_div[['ticker', 'ex_date', 'cash_dividend']]

            dbg.log(f"[{ticker}] ✅ [yfinance] 備援抓取完成！")
            return df_div, df_earn
        except Exception as e:
            dbg.error(f"[{ticker}] ❌ [yfinance] 備援抓取亦失敗: {e}")
            return pd.DataFrame(), pd.DataFrame()