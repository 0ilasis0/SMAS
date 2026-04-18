import json
from datetime import datetime
from pathlib import Path

from data.const import MacroTicker, TimeUnit
from data.event_fetcher import TWSEEventFetcher
from data.fetcher import Fetcher
from data.manager import DataManager
from data.params import DataLimit
from debug import dbg
from path import PathConfig


class DataUpdater:
    """
    資料更新專員 (獨立運行版)
    負責管理每日資料抓取邏輯與更新快取檔紀錄。
    """
    def __init__(self, db: DataManager, fetcher: Fetcher):
        self.db = db
        self.fetcher = fetcher
        self.cache_file = Path(PathConfig.CACHE_FILE)

    def _needs_update(self, cache_key: str) -> bool:
        today_str = datetime.now().strftime('%Y-%m-%d')
        if not self.cache_file.exists(): return True
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            return cache.get(cache_key) != today_str
        except Exception:
            return True

    def _mark_updated(self, cache_key: str):
        today_str = datetime.now().strftime('%Y-%m-%d')
        cache = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception: pass
        cache[cache_key] = today_str
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            dbg.war(f"無法寫入更新快取檔: {e}")

    def update_market_data(self, ticker: str, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR, force_wipe: bool = False, force_sync: bool = False) -> bool:
        if force_wipe:
            dbg.log(f"🧹 [資料清洗] 正在清空 {ticker} 舊版歷史資料庫...")
            self.db.clear_ticker_data(ticker)

        success = True

        # ================== 1. 個股更新 ==================
        if force_wipe or force_sync or self._needs_update(ticker):
            dbg.log(f"[{ticker}] 正在從網路更新個股歷史資料...")
            daily_df = self.fetcher.fetch_daily_data(ticker, period=period, unit=unit)
            if not daily_df.empty:
                self.db.save_daily_data(ticker, daily_df)
                self._mark_updated(ticker)
                dbg.log(f"[{ticker}] 資料庫更新成功！")
            else:
                dbg.error(f"[{ticker}] 抓取資料失敗，請檢查網路。")
                success = False
        else:
            dbg.log(f"⚡ [{ticker}] 今日已同步過最新資料，跳過網路抓取。")

        # ================== 2. 大盤更新 ==================
        for macro_item in MacroTicker:
            m_ticker = macro_item.value
            if force_wipe or force_sync or self._needs_update(m_ticker):
                dbg.log(f"[{m_ticker}] 正在同步更新大盤/總經資料...")
                df_macro = self.fetcher.fetch_daily_data(m_ticker, period=period, unit=unit)
                if not df_macro.empty:
                    self.db.save_daily_data(m_ticker, df_macro)
                    self._mark_updated(m_ticker)
                else:
                    dbg.war(f"[{m_ticker}] 總經資料更新失敗。")

        # ================== 3. 企業事件更新 ==================
        event_cache_key = "corporate_events_twse"
        if force_wipe or force_sync or self._needs_update(event_cache_key):
            dbg.log("[事件日曆] 正在同步證交所除權息與法說會資料...")
            event_fetcher = TWSEEventFetcher()

            df_div = event_fetcher.fetch_upcoming_dividends()
            if not df_div.empty: self.db.save_dividends_calendar(df_div)

            df_earn = event_fetcher.fetch_upcoming_earnings()
            if not df_earn.empty: self.db.save_earnings_calendar(df_earn)

            self._mark_updated(event_cache_key)
        else:
            dbg.log("⚡ [事件日曆] 今日已同步過證交所事件，跳過網路抓取。")

        return success