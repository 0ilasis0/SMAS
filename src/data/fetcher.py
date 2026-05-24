import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from base import MathTool
from data.const import StockCol, TimeUnit, YfInterval
from data.params import DataLimit
from debug import dbg

try:
    _ = yf.Ticker("SPY").history(period="1d")
except Exception:
    pass

class Fetcher:

    INTRADAY_INDEX = 'Datetime'
    address = "Asia/Taipei"

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

        return self._process_fetched_data(df, ticker_symbol, index_name=StockCol.DATE)

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

        return self._process_fetched_data(df, ticker_symbol, index_name=self.INTRADAY_INDEX)

    def _process_fetched_data(self, df: pd.DataFrame, ticker_symbol: str, index_name: str) -> pd.DataFrame:
        """
        共用資料處理管線：負責欄位改名、對齊、補零與時區校正。
        """
        if df.empty:
            dbg.war(f"[{ticker_symbol}] 抓取失敗或無資料")
            return pd.DataFrame()

        df.columns = df.columns.str.lower()
        df.rename(columns={'adj close': StockCol.ADJ_CLOSE}, inplace=True)

        # 防呆：如果 yfinance 沒有給 adj close，就用 close 代替
        if StockCol.ADJ_CLOSE not in df.columns:
            df[StockCol.ADJ_CLOSE] = df[StockCol.CLOSE] if StockCol.CLOSE in df.columns else 0.0

        # 確保 OHLCV 欄位順序正確
        expected_cols = StockCol.get_ohlcv()
        df = df.reindex(columns=expected_cols).fillna(0)

        df.index.name = index_name

        # 精準時區校正：轉為當地時間並拔除時區標籤
        if df.index.tz is not None:
            df.index = df.index.tz_convert(self.address).tz_localize(None)

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
                dbg.war(f"抓取發生例外錯誤: {e} (嘗試 {attempt + 1}/{self.MAX_RETRIES}):\n{traceback.format_exc()}")

            if attempt < self.MAX_RETRIES - 1:
                sleep_time = self.BACKOFF_FACTOR ** attempt
                dbg.log(f"等待 {sleep_time} 秒後重試...")
                time.sleep(sleep_time)
            else:
                dbg.error("已達最大重試次數，放棄抓取。")

        return pd.DataFrame()

    def fetch_foreign_futures_oi(self, days: int = DataLimit.DAILY_MAX_YEAR * 365) -> pd.DataFrame:
        """
        [宏觀升級] 抓取外資「台指期 (TX) 未平倉淨部位 (Net Open Interest)」。
        負數代表外資滿手空單，是台股崩盤的最強領先指標。
        來源: FinMind Open API
        """
        dbg.log("正在透過 FinMind API 抓取外資台指期未平倉數據...")

        # 計算回推的起始日期
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanFuturesInstitutionalInvestors",
            "data_id": "TX", # TX 代表大台指期貨
            "start_date": start_date
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("msg") == "success" and len(data.get("data", [])) > 0:
                    df = pd.DataFrame(data["data"])
                    df_foreign = df[df['institutional_investors'].str.contains('外資|Foreign', case=False, na=False)].copy()

                    if df_foreign.empty:
                        investor_list = df['institutional_investors'].unique().tolist()
                        dbg.war(f"FinMind 回傳的投資人類別有: {investor_list}")
                        dbg.war("FinMind API 未回傳包含『外資』的期貨數據，請檢查上方印出的類別名稱。")
                        return pd.DataFrame()

                    # 計算「未平倉淨部位 (多單 - 空單)」
                    df_foreign['futures_net_oi'] = (
                        df_foreign['long_open_interest_balance_volume'] -
                        df_foreign['short_open_interest_balance_volume']
                    )

                    # 格式化為與系統相容的 DataFrame
                    df_foreign['date'] = pd.to_datetime(df_foreign['date'])
                    df_foreign.set_index('date', inplace=True)
                    df_foreign.index.name = StockCol.DATE

                    # 只保留我們需要的核心欄位
                    df_clean = df_foreign[['futures_net_oi']].copy()

                    dbg.log(f"成功抓取外資期貨 OI 數據，共 {len(df_clean)} 筆。")
                    return df_clean

            except Exception as e:
                dbg.war(f"FinMind API 抓取失敗 (嘗試 {attempt + 1}/{self.MAX_RETRIES}): {e}")

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_FACTOR ** attempt)

        dbg.error("外資期貨 OI 抓取已達最大重試次數，放棄。")
        return pd.DataFrame()
