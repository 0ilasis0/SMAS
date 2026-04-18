import json
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
        self.session = requests.Session()

        # 幫爬蟲戴上 Chrome 瀏覽器的面具，避免被證交所的防火牆擋下
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

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
            resp = self.session.get(self.DIVIDEND_URL, timeout=10)
            resp.raise_for_status()

            # 防呆，確認回傳的不是空字串或網頁檔
            if not resp.text.strip() or "<html>" in resp.text.lower():
                dbg.war("TWSE 除權息 API 回傳空白或網頁 (可能正在維護中)。")
                return pd.DataFrame()

            data = resp.json()
            results = []
            for item in data:
                ticker = item.get('Code', '').strip()
                if not ticker: continue

                results.append({
                    'ticker': f"{ticker}.TW",
                    'ex_date': self._parse_roc_date(item.get('Date', '')),
                    'cash_dividend': float(item.get('CashDividend') or 0.0)
                })

            df = pd.DataFrame(results)
            if not df.empty:
                df = df[df['ex_date'] != ""].drop_duplicates()
                dbg.log(f"成功抓取 {len(df)} 筆即將到來的除權息事件。")
            return df
        except json.JSONDecodeError:
            dbg.war("TWSE 回傳資料格式錯誤 (非 JSON 格式)，可能是週末維護中。")
            return pd.DataFrame()
        except Exception as e:
            dbg.error(f"除權息資料更新異常: {e}")
            return pd.DataFrame()

    def fetch_upcoming_earnings(self) -> pd.DataFrame:
        dbg.log("正在從 TWSE 抓取最新法說會日程數據...")
        try:
            resp = self.session.get(self.EARNINGS_URL, timeout=10)
            resp.raise_for_status()

            if not resp.text.strip() or "<html>" in resp.text.lower():
                dbg.war("TWSE 法說會 API 回傳空白或網頁 (可能正在維護中)。")
                return pd.DataFrame()

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
        except json.JSONDecodeError:
            dbg.war("TWSE 回傳資料格式錯誤 (非 JSON 格式)，可能是週末維護中。")
            return pd.DataFrame()
        except Exception as e:
            dbg.error(f"法說會資料更新異常: {e}")
            return pd.DataFrame()