import time

import pandas as pd
import yfinance as yf

from base import MathTool
from data.const import StockCol, TimeUnit, YfInterval
from data.params import DataLimit
from debug import dbg


class Fetcher:

    INTRADAY_INDEX = 'Datetime'

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2

    def fetch_daily_data(self, ticker_symbol: str, period: int, unit: str) -> pd.DataFrame:
        """
        抓取指定標的的中長期日 K 線歷史資料。
        """
        if unit == TimeUnit.YEAR:
            valid_period = MathTool.clamp(period, 1, DataLimit.DAILY_MAX_YEAR)
        elif unit == TimeUnit.MONTH:
            valid_period = MathTool.clamp(period, 1, DataLimit.DAILY_MAX_MONTH)
        else:
            dbg.war("時間單位輸入錯誤")
            return pd.DataFrame()

        ticker = yf.Ticker(ticker_symbol)

        df = self._safe_fetch(
            ticker,
            period=f"{valid_period}{unit}",
            interval=f"{YfInterval.DAILY}",
            auto_adjust=False,
            actions=False
        )

        if df.empty:
            dbg.war(f"抓取失敗或無資料: {ticker_symbol}")
            return pd.DataFrame()

        df.columns = df.columns.str.lower()
        df.rename(columns={'adj close': StockCol.ADJ_CLOSE}, inplace=True)

        expected_cols = StockCol.get_ohlcv()
        df = df.reindex(columns=expected_cols).fillna(0)

        df.index.name = StockCol.DATE

        # 精準時區校正：先轉為台灣時間，再拔除標籤
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Taipei').tz_localize(None)

        return df

    def fetch_intraday_data(self, ticker_symbol: str, days: int = DataLimit.INTRADAY_MAX_DAY) -> pd.DataFrame:
        """
        抓取指定標的的分時資料 (預設 5 分鐘 K 線)。
        """
        valid_days = MathTool.clamp(days, 1, DataLimit.INTRADAY_MAX_DAY)

        ticker = yf.Ticker(ticker_symbol)

        df = self._safe_fetch(
            ticker,
            period=f"{valid_days}{TimeUnit.DAY}",
            interval=f"{YfInterval.INTRADAY_5M}",
            auto_adjust=False,
            actions=False
        )

        if df.empty:
            dbg.war(f"[{ticker_symbol}] 分時資料抓取失敗或無資料")
            return pd.DataFrame()

        df.columns = df.columns.str.lower()

        df.rename(columns={'adj close': StockCol.ADJ_CLOSE}, inplace=True)
        if StockCol.ADJ_CLOSE not in df.columns:
            df[StockCol.ADJ_CLOSE] = df[StockCol.CLOSE]

        expected_cols = StockCol.get_ohlcv()
        df = df.reindex(columns=expected_cols).fillna(0)

        df.index.name = self.INTRADAY_INDEX

        # 精準時區校正
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Taipei').tz_localize(None)

        return df

    def _safe_fetch(self, ticker: yf.Ticker, **kwargs) -> pd.DataFrame:
        """
        核心抓取引擎：內建 Exponential Backoff (指數退避) 重試機制
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                df = ticker.history(**kwargs)
                if not df.empty: return df

                dbg.war(f"抓取回傳空資料，可能無資料或遭限流 (嘗試 {attempt + 1}/{self.MAX_RETRIES})")
            except Exception as e:
                dbg.war(f"抓取發生例外錯誤: {e} (嘗試 {attempt + 1}/{self.MAX_RETRIES})")

            if attempt < self.MAX_RETRIES - 1:
                sleep_time = self.BACKOFF_FACTOR ** attempt
                dbg.log(f"等待 {sleep_time} 秒後重試...")
                time.sleep(sleep_time)

        dbg.error("已達最大重試次數，放棄抓取。")
        return pd.DataFrame()