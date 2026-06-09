import json
from datetime import datetime, timedelta
from pathlib import Path

from data.const import MacroTicker, TimeUnit
from data.event_fetcher import HybridEventFetcher
from data.macro_fetcher import MacroFetcher
from data.manager import DataManager
from data.params import DataLimit
from data.stock_fetcher import StockFetcher
from debug import dbg
from ml.const import MacroDbKey
from path import PathConfig


class DataUpdater:
    """
    資料更新專員 (獨立運行版)
    負責管理每日資料抓取邏輯與更新快取檔紀錄。
    """
    def __init__(self,
                 db: DataManager,
                 stock_fetcher: StockFetcher = None,
                 macro_fetcher: MacroFetcher = None,
                 event_fetcher: HybridEventFetcher = None
                ):
        self.db = db
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.macro_fetcher = macro_fetcher or MacroFetcher()
        self.event_fetcher = event_fetcher or HybridEventFetcher()
        self.cache_file = Path(PathConfig.CACHE_FILE)

    def update_market_data(self, ticker: str, period: int = DataLimit.DAILY_DEFAULT_YEAR, unit: TimeUnit = TimeUnit.YEAR, force_wipe: bool = False, force_sync: bool = False) -> bool:
        if force_wipe:
            dbg.log(f"🧹 [資料清洗] 正在清空 {ticker} 舊版歷史資料庫...")
            self.db.clear_ticker_data(ticker)

        success = True

        # ================== 1. 個股更新 ==================
        if force_wipe or force_sync or self._needs_update(ticker):
            dbg.log(f"[{ticker}] 正在從網路更新個股歷史資料...")
            daily_df = self.stock_fetcher.fetch_daily_data(ticker, period=period, unit=unit)
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
                df_macro = self.stock_fetcher.fetch_daily_data(m_ticker, period=period, unit=unit)
                if not df_macro.empty:
                    self.db.save_daily_data(m_ticker, df_macro)
                    self._mark_updated(m_ticker)
                else:
                    dbg.war(f"[{m_ticker}] 總經資料更新失敗。")
                    success = False

        # ================== 3. 企業事件更新 ==================
        event_cache_key = f"events_{ticker}"
        if force_wipe or self._needs_update(event_cache_key):
            dbg.log(f"[{ticker}] [事件日曆] 正在同步除權息與法說會資料...")

            # 傳入 ticker，同時取得除權息與法說會 DataFrame
            df_div, df_earn = self.event_fetcher.fetch_upcoming_events(ticker)

            if not df_div.empty:
                self.db.save_dividends_calendar(df_div)
            if not df_earn.empty:
                self.db.save_earnings_calendar(df_earn)

            self._mark_updated(event_cache_key)
        else:
            dbg.log(f"⚡ [{ticker}] [事件日曆] 今日已同步過該股事件，跳過網路抓取。")

        # ================== 4. 宏觀與籌碼更新 (四大黑天鵝防禦指標完全體) ==================
        macro_features = {
            "macro_futures_oi": {
                "name": "外資台指期未平倉淨部位",
                "fetch_func": self.macro_fetcher.fetch_foreign_futures_oi,
                "db_key": MacroDbKey.FUTURES_OI
            },
            "macro_retail_ls": {
                "name": "散戶小台多空比",
                "fetch_func": self.macro_fetcher.fetch_retail_ls_ratio,
                "db_key": MacroDbKey.RETAIL_LS_RATIO
            },
            "macro_pc_ratio": {
                "name": "選擇權 Put/Call Ratio",
                "fetch_func": self.macro_fetcher.fetch_options_pc_ratio,
                "db_key": MacroDbKey.PC_RATIO_CLOSE
            },
            "macro_adl_value": {
                "name": "騰落指標 (ADL 替代: 櫃買指數)",
                "fetch_func": self.macro_fetcher.fetch_twse_adl_value,
                "db_key": MacroDbKey.ADL_VALUE
            }
        }

        # 取得前一個交易日的大致基準（通常是昨天或前天）
        target_fresh_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        for cache_key, config in macro_features.items():
            # 🔍 第一重檢查：從 .db 資料庫實體讀取歷史 Series
            db_series = self.db.get_macro_data(config["db_key"])

            db_is_fresh = False
            latest_db_date = "無"
            if not db_series.empty:
                # 檢索資料庫裡最新一筆資料的日期
                latest_db_date = db_series.index[-1].strftime('%Y-%m-%d')
                if latest_db_date >= target_fresh_date:
                    db_is_fresh = True

            # 🔍 第二重檢查：結合 force 參數與資料庫新鮮度
            should_fetch = force_wipe or force_sync or (not db_is_fresh)

            if should_fetch:
                # 🛡️ 第三重檢查安全網：如果這趟批次中「剛剛已經試過且蓋章了」，先跳過
                if not force_sync and not self._needs_update(cache_key):
                    continue

                dbg.log(f"[宏觀防禦] 正在從網路同步更新 {config['name']} ...")
                df_macro_feat = config["fetch_func"]()

                if not df_macro_feat.empty:
                    # 🟢 真正抓到資料：寫入資料庫並更新 JSON 快取日期
                    self.db.save_macro_data(config["db_key"], df_macro_feat)
                    self._mark_updated(cache_key)
                else:
                    # ❌ 真正沒抓到：印出警告，但「依然在 JSON 蓋章」防止當日無限迴圈
                    dbg.war(f"[宏觀防禦] {config['name']} 本次同步失敗。將在下一趟任務重試。")
                    self._mark_updated(cache_key)

        return success

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