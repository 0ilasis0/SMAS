import os
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from base import MathTool
from const import GlobalCol
from data.const import StockCol, TimeUnit, YfInterval
from data.params import DataLimit
from debug import dbg
from path import PathConfig

try:
    _ = yf.Ticker("SPY").history(period="1d")
except Exception:
    pass

class Fetcher:

    INTRADAY_INDEX = 'Datetime'
    address = "Asia/Taipei"

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2

    def __init__(self) -> None:
        load_dotenv(PathConfig.ENV_FILE)
        self.finmind_token = os.getenv(GlobalCol.FINMIND_API_KEYS)

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

    def fetch_intraday_data(self, ticker_symbol: str, days: int = DataLimit.INTRADAY_DEFAULT_DAY) -> pd.DataFrame:
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

                history_metadata = getattr(ticker, '_history', None)
                last_error = "未知 (yfinance 未拋出標準異常)"

                if history_metadata and hasattr(history_metadata, 'errors'):
                    last_error = history_metadata.errors

                dbg.war(f"❌ [yfinance 偵錯] 伺服器回傳空資料！(嘗試 {attempt + 1}/{self.MAX_RETRIES})")
                dbg.war(f"   👉 內部可能原因: {last_error}")

                try:
                    info_keys = list(ticker.info.keys())[:3] if ticker.info else []
                    dbg.log(f"   ℹ️ Ticker 狀態確認: 代號存在且可連線，部分欄位快照: {info_keys}")
                except Exception as info_err:
                    dbg.error(f"   🚨 Ticker 狀態確認失敗 (可能代號不合法或 IP 遭封鎖): {info_err}")

            except Exception as e:
                dbg.war(f"抓取發生例外錯誤: {e} (嘗試 {attempt + 1}/{self.MAX_RETRIES}):\n{traceback.format_exc()}")

            if attempt < self.MAX_RETRIES - 1:
                sleep_time = self.BACKOFF_FACTOR ** attempt
                dbg.log(f"等待 {sleep_time} 秒後重試...")
                time.sleep(sleep_time)
            else:
                dbg.error("已達最大重試次數，放棄抓取。")

        return pd.DataFrame()

    def fetch_twse_adl_value(self, days: int = DataLimit.DAILY_RETAIL_LS_YEAR * 365) -> pd.DataFrame:
        """
        [防禦升級] 利用 FinMind 的「櫃買報酬指數 (TPEx)」完美替代傳統 ADL 騰落指標。
        🟢 終極破案版：徹底拋棄不穩定的 yfinance，改用 FinMind 官方免費、零時區 Bug 的報酬指數表！
        來源: FinMind Open API (TaiwanStockTotalReturnIndex)
        """
        dbg.log("正在透過 FinMind API 獲取櫃買報酬指數 (TPEx) 建立大盤廣度背離指標...")

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data"

        params = {
            "dataset": "TaiwanStockTotalReturnIndex",  # 官方正宗大盤報酬指數資料集
            "data_id": "TPEx",                         # 精準指定：櫃買報酬指數
            "start_date": start_date,
            "token": self.finmind_token
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                # 取得資料陣列
                data_list = response.json().get("data", [])

                if not data_list:
                    continue

                df = pd.DataFrame(data_list)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.index.name = StockCol.DATE

                # 確保收盤價欄位存在 (FinMind 此資料集通常將指數數值放在 'price' 欄位)
                price_col = 'price' if 'price' in df.columns else 'close'

                if price_col not in df.columns:
                    dbg.error(f"❌ TPEx 資料集缺失價格欄位，現有欄位: {df.columns.tolist()}")
                    return pd.DataFrame()

                # 將數值重新命名為 'adl_value'，完美對接後端的特徵工程
                df_res = pd.DataFrame(index=df.index)
                df_res['adl_value'] = df[price_col].astype(float)

                dbg.log(f"✅ 成功載入櫃買報酬指數 (TPEx)，共 {len(df_res)} 筆交易日。")
                return df_res[['adl_value']]

            except Exception as e:
                dbg.war(f"FinMind TPEx 指數抓取失敗 (嘗試 {attempt + 1}/{self.MAX_RETRIES}): {e}")

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_FACTOR ** attempt)

        dbg.error("❌ 櫃買報酬指數 (TPEx) 連線重試耗盡，放棄抓取。")
        return pd.DataFrame()

    def fetch_foreign_futures_oi(self, days: int = DataLimit.DAILY_DEFAULT_YEAR * 365) -> pd.DataFrame:
        """
        抓取外資「台指期 (TX) 未平倉淨部位 (Net Open Interest)」。
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


