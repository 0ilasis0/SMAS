import json
from datetime import datetime
from typing import TYPE_CHECKING

from data.const import MacroTicker, TimeUnit
from data.params import DataLimit
from debug import dbg

if TYPE_CHECKING:
    from .core import QuantAIEngine

class DataUpdater:
    """
    [模組 1] 資料更新專員
    負責管理每日資料抓取邏輯與更新快取檔紀錄。
    """
    def __init__(self, engine):
        self.engine: "QuantAIEngine" = engine

    def _needs_update(self, ticker: str) -> bool:
        """檢查今天是否已經更新過該標的"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        if not self.engine.cache_file.exists():
            return True
        try:
            with open(self.engine.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            # 如果紀錄的日期不是今天，就代表需要更新
            return cache.get(ticker) != today_str
        except Exception:
            return True

    def _mark_updated(self, ticker: str):
        """標記該標的今天已經成功更新"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        cache = {}
        if self.engine.cache_file.exists():
            try:
                with open(self.engine.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception:
                pass

        cache[ticker] = today_str
        try:
            with open(self.engine.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            dbg.war(f"無法寫入更新快取檔: {e}")

    def update_market_data(self, period: int = DataLimit.DAILY_MAX_YEAR, unit: TimeUnit = TimeUnit.YEAR, force_wipe: bool = False) -> bool:
        """從網路抓取最新歷史資料並寫入資料庫"""
        ticker = self.engine.config.ticker

        if force_wipe:
            dbg.log(f"🧹 [資料清洗] 正在清空 {ticker} 舊版歷史資料庫...")
            self.engine.db.clear_ticker_data(ticker)

        success = True

        # 1. 個股更新邏輯
        if force_wipe or self._needs_update(ticker):
            dbg.log(f"[{ticker}] 正在從網路更新個股歷史資料...")
            daily_df = self.engine.fetcher.fetch_daily_data(ticker, period=period, unit=unit)

            if not daily_df.empty:
                self.engine.db.save_daily_data(ticker, daily_df)
                self._mark_updated(ticker)  # 成功才標記
                dbg.log(f"[{ticker}] 資料庫更新成功！")
            else:
                dbg.error(f"[{ticker}] 抓取資料失敗，請檢查網路。")
                success = False
        else:
            dbg.log(f"⚡ [{ticker}] 今日已同步過最新資料，跳過網路抓取。")

        # 2. 大盤/總經更新邏輯 (遍歷 MacroTicker Enum)
        for macro_item in MacroTicker:
            m_ticker = macro_item.value
            if force_wipe or self._needs_update(m_ticker):
                dbg.log(f"[{m_ticker}] 正在同步更新大盤/總經資料...")
                df_macro = self.engine.fetcher.fetch_daily_data(m_ticker, period=period, unit=unit)

                if not df_macro.empty:
                    self.engine.db.save_daily_data(m_ticker, df_macro)
                    self._mark_updated(m_ticker)
                else:
                    dbg.war(f"[{m_ticker}] 總經資料更新失敗，可能被 Yahoo 阻擋或無數據。")
            else:
                pass # 保持 Terminal 乾淨

        return success