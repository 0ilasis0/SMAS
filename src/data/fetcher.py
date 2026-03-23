import pandas as pd
import yfinance as yf
from base import MathTool
from data.variable import DataLimit, StockCol, TimeUnit, YfInterval
from debug import dbg


class Fetcher:

    DAILY_INDEX = StockCol.DATE
    INTRADAY_INDEX = 'Datetime'
    CATALOG = [StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.CLOSE, StockCol.VOLUME]

    def fetch_daily_data(self, ticker_symbol: str, period_length: int, unit: str = TimeUnit.YEAR) -> pd.DataFrame:
        """
        抓取指定標的的中長期日 K 線歷史資料。
        """
        if unit == TimeUnit.YEAR:
            valid_period = MathTool.clamp(period_length, 1, DataLimit.DAILY_MAX_YEAR)
        elif unit == TimeUnit.MONTH:
            valid_period = MathTool.clamp(period_length, 1, DataLimit.DAILY_MAX_MONTH)
        else:
            dbg.war("時間單位輸入錯誤")
            return pd.DataFrame()

        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=f"{valid_period}{unit}", interval=f"{YfInterval.DAILY}")

        if df.empty:
            dbg.war(f"抓取失敗或無資料: {ticker_symbol}")
            return pd.DataFrame()

        df = df[self.CATALOG]
        df.index.name = self.DAILY_INDEX

        return df

    def fetch_intraday_data(self, ticker_symbol: str, days: int = DataLimit.INTRADAY_MAX_DAY) -> pd.DataFrame:
        """
        抓取指定標的的分時資料 (5 分鐘 K 線)。
        """
        valid_days = MathTool.clamp(days, 1, DataLimit.INTRADAY_MAX_DAY)

        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=f"{valid_days}{TimeUnit.DAY}", interval=f"{YfInterval.INTRADAY_5M}")

        if df.empty:
            dbg.war("抓取失敗或無資料")
            return pd.DataFrame()

        df = df[self.CATALOG]
        df.index.name = self.INTRADAY_INDEX

        return df
