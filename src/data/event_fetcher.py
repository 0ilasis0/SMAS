from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from debug import dbg


class TWSEEventFetcher:
    """
    從台灣證券交易所 (TWSE) OpenAPI 抓取企業事件資料 (僅含上市)。
    提供除權息預告與法人說明會 (法說會) 日程。
    """
    DIVIDEND_URL = "https://openapi.twse.com.tw/v1/exchangeReport/TWT48U"
    EARNINGS_URL = "https://openapi.twse.com.tw/v1/company/investorConference"

    def __init__(self):
        # 建立帶有自動重試機制的 Session
        self.session = requests.Session()
        # 設定重試 3 次，遇到 500, 502, 503, 504 時自動退避重試
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 重試間隔：1s, 2s, 4s...
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _parse_roc_date(self, roc_date_str: str) -> str:
        """將民國或西元日期字串轉換為標準 YYYY-MM-DD。"""
        try:
            clean_date = roc_date_str.replace("/", "").replace("-", "").strip()
            if not clean_date or len(clean_date) < 6:
                return ""

            year_part = clean_date[:-4]
            month_day_part = clean_date[-4:]

            year_int = int(year_part)
            # 自動判斷西元與民國
            ce_year = year_int if year_int > 1000 else year_int + 1911

            date_str = f"{ce_year}-{month_day_part[:2]}-{month_day_part[2:]}"

            # 利用 datetime 驗證日期真實性 (避免出現 2月30日 等鬼數據)
            datetime.strptime(date_str, "%Y-%m-%d")

            return date_str
        except Exception as e:
            dbg.war(f"日期解析失敗 ({roc_date_str}): {e}")
            return ""

    def fetch_upcoming_dividends(self) -> pd.DataFrame:
        dbg.log("正在從 TWSE 抓取最新除權息預告數據...")
        try:
            # 使用配置好的 session 發送請求
            resp = self.session.get(self.DIVIDEND_URL, timeout=10)
            resp.raise_for_status() # 檢查 4xx/5xx 錯誤

            data = resp.json()
            results = []
            for item in data:
                ticker = item.get('Code', '').strip()
                if not ticker: continue

                results.append({
                    'ticker': f"{ticker}.TW",
                    'ex_date': self._parse_roc_date(item.get('Date', '')),
                    # 🚀 升級：增強數值轉換的防呆，防範空字串轉換 float 報錯
                    'cash_dividend': float(item.get('CashDividend') or 0.0)
                })

            df = pd.DataFrame(results)
            if not df.empty:
                df = df[df['ex_date'] != ""].drop_duplicates()
                dbg.log(f"成功抓取 {len(df)} 筆即將到來的除權息事件。")
            return df
        except Exception as e:
            dbg.error(f"除權息資料更新異常: {e}")
            return pd.DataFrame()

    def fetch_upcoming_earnings(self) -> pd.DataFrame:
        dbg.log("正在從 TWSE 抓取最新法說會日程數據...")
        try:
            resp = self.session.get(self.EARNINGS_URL, timeout=10)
            resp.raise_for_status()

            data = resp.json()
            results = []
            for item in data:
                ticker = item.get('Code', '').strip()
                if not ticker: continue

                results.append({
                    'ticker': f"{ticker}.TW",
                    'earnings_date': self._parse_roc_date(item.get('Date', ''))
                })

            df = pd.DataFrame(results)
            if not df.empty:
                df = df[df['earnings_date'] != ""].drop_duplicates()
                dbg.log(f"成功抓取 {len(df)} 筆即將到來的法說會日期。")
            return df
        except Exception as e:
            dbg.error(f"法說會資料更新異常: {e}")
            return pd.DataFrame()