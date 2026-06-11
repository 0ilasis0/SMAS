import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import MacroDbKey, MacroRawCol
from path import PathConfig


class DataManager:
    def __init__(self, db_path: str = PathConfig.IDSS_DATA):
        self.db_path = db_path
        self.setup()

    def setup(self):
        db_path_obj = Path(self.db_path)
        db_dir = db_path_obj.parent

        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            dbg.log(f"已自動建立資料庫目錄: {db_dir}")

        self._create_tables()

    def _create_tables(self):
        """初始化資料表 (Table Schema)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 日線表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_k_lines (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    PRIMARY KEY (ticker, date)
                )
            ''')

            # 分時線表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_k_lines (
                    ticker TEXT,
                    datetime TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    PRIMARY KEY (ticker, datetime)
                )
            ''')

            # 建立自選股表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_watchlist (
                    ticker TEXT PRIMARY KEY,
                    added_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dividends_calendar (
                    ticker TEXT,
                    ex_date TEXT,
                    cash_dividend REAL
                )
            ''')
            # 建立索引加快查詢速度
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_div_ticker ON dividends_calendar(ticker)')

            # 法說會日程表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS earnings_calendar (
                    ticker TEXT,
                    earnings_date TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS macro_time_series (
                    metric_name TEXT,
                    date TEXT,
                    value REAL,
                    PRIMARY KEY (metric_name, date)
                )
            ''')

            # 為了加速未來 get_aligned_market_data 時的 Join 操作
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings_calendar(ticker)')

            conn.commit()

    def clear_ticker_data(self, ticker: str):
        """刪除特定標的的所有歷史日線與分時資料。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM daily_k_lines WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM intraday_k_lines WHERE ticker = ?", (ticker,))
            conn.commit()
            dbg.log(f"已徹底清空 {ticker} 的歷史資料快取。")

    def save_daily_data(self, ticker: str, df: pd.DataFrame):
        """將日 K 線 DataFrame 存入 SQLite (修復 UPSERT 陷阱)"""
        if df.empty: return

        df_save = df.copy()
        df_save = df_save.dropna(subset=[StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.CLOSE])
        if df_save.empty: return

        if StockCol.ADJ_CLOSE not in df_save.columns:
            df_save[StockCol.ADJ_CLOSE] = df_save[StockCol.CLOSE]

        records = []
        for index, row in df_save.iterrows():
            date_str = str(index).split(' ')[0]
            vol = int(row[StockCol.VOLUME]) if pd.notna(row[StockCol.VOLUME]) else 0
            records.append((
                ticker, date_str,
                row[StockCol.OPEN], row[StockCol.HIGH], row[StockCol.LOW],
                row[StockCol.CLOSE], vol, row[StockCol.ADJ_CLOSE]
            ))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO daily_k_lines
                (ticker, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            dbg.log(f"成功儲存 {ticker} 日 K 線資料，共 {len(records)} 筆。")

    def save_intraday_data(self, ticker: str, df: pd.DataFrame):
        """將分時 K 線 DataFrame 存入 SQLite"""
        if df.empty: return

        df_save = df.copy()
        df_save.columns = [str(c).strip().lower() for c in df_save.columns]
        df_save['ticker'] = ticker

        df_save = df_save.reset_index()
        df_save = df_save.rename(columns={df_save.columns[0]: 'datetime'})
        df_save['datetime'] = df_save['datetime'].astype(str) # 保留完整時間字串

        with sqlite3.connect(self.db_path) as conn:
            df_save.to_sql('intraday_k_lines', conn, if_exists='append', index=False)
            dbg.log(f"成功儲存 {ticker} 分時 K 線資料，共 {len(df_save)} 筆。")

    def add_to_watchlist(self, ticker: str):
        """新增標的至自選股清單"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO user_watchlist (ticker) VALUES (?)
            ''', (ticker,))
            conn.commit()
            dbg.log(f"已將 {ticker} 加入自選清單。")

    def get_watchlist(self) -> list[str]:
        """讀取使用者的自選股清單"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker FROM user_watchlist ORDER BY added_time ASC")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def remove_from_watchlist(self, ticker: str):
        """從自選股清單移除標的"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_watchlist WHERE ticker = ?", (ticker,))
            conn.commit()
            dbg.log(f"已將 {ticker} 從自選清單移除。")

    def get_aligned_market_data(self, stock_ticker: str, macro_tickers: list[str]) -> pd.DataFrame:
        """
        機構級數據對齊引擎 (支援宏觀期貨數據)
        """
        df_stock = self.get_daily_data(stock_ticker)

        if df_stock.empty: return df_stock

        aligned_df = df_stock.copy()
        overseas_tickers = MacroTicker.get_overseas_tickers()
        macro_cols = [] # 記錄所有加入的大盤欄位

        # 1. 處理一般的 Yahoo Finance 總經指數 (VIX, 美債, 費半等)
        for mt in macro_tickers:
            df_macro = self.get_daily_data(mt)
            if df_macro.empty: continue

            prefix = mt.replace('^', '') + "_"
            df_macro = df_macro.add_prefix(prefix)
            macro_cols.extend(df_macro.columns.tolist())

            if mt in overseas_tickers:
                df_macro_ffilled = df_macro.asfreq('D', method='ffill')
                df_macro_shifted = df_macro_ffilled.shift(1)
                aligned_df = aligned_df.join(df_macro_shifted, how='left')
            else:
                aligned_df = aligned_df.join(df_macro, how='left')

        # 2. 將五大獨立的宏觀籌碼指標加入對齊
        # 定義 DB Key 與 對應的 DataFrame 欄位名稱 (必須與 MarketFeatureEngine 一致)
        chip_features = {
            MacroDbKey.FUTURES_OI: MacroRawCol.FUTURES_NET_OI,
            MacroDbKey.RETAIL_LS_RATIO: MacroRawCol.RETAIL_LS_RATIO,
            MacroDbKey.PC_RATIO_CLOSE: MacroRawCol.PC_RATIO_CLOSE,
            MacroDbKey.ADL_VALUE: MacroRawCol.ADL_VALUE,
        }

        for db_key, col_name in chip_features.items():
            # 從資料庫提取出該指標的 Series
            s_chip = self.get_macro_data(db_key)
            if not s_chip.empty:
                # 將 Series 轉為 DataFrame，並指定我們需要的欄位名稱
                df_chip = s_chip.to_frame(name=col_name)
                macro_cols.append(col_name)

                # 台股本土籌碼與大盤同步開盤，直接使用 left join，無需 shift
                aligned_df = aligned_df.join(df_chip, how='left')
            else:
                dbg.war(f"對齊引擎未能在 DB 找到籌碼指標 [{db_key}]")

        # 3. 統一填補所有宏觀資料的假日期缺口 (例如國定假日)
        if macro_cols:
            aligned_df[macro_cols] = aligned_df[macro_cols].ffill()

        return aligned_df

    def get_daily_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        return self._fetch_data(
            table_name='daily_k_lines', time_col='date', ticker=ticker, start_time=start_date, end_time=end_date
        )

    def get_intraday_data(self, ticker: str, start_datetime: str = None, end_datetime: str = None) -> pd.DataFrame:
        return self._fetch_data(
            table_name='intraday_k_lines', time_col='datetime', ticker=ticker, start_time=start_datetime, end_time=end_datetime
        )

    def _fetch_data(self, table_name: str, time_col: str, ticker: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} WHERE ticker = ?"
        params = [ticker]

        if start_time:
            query += f" AND {time_col} >= ?"
            params.append(start_time)
        if end_time:
            query += f" AND {time_col} <= ?"
            params.append(end_time)

        query += f" ORDER BY {time_col}"

        with sqlite3.connect(self.db_path) as conn:
            # 讓 pandas 自動解析日期，並將其設為 index
            df = pd.read_sql_query(query, conn, params=params, index_col=time_col, parse_dates=[time_col])

        if not df.empty:
            if StockCol.TICKER in df.columns:
                df = df.drop(columns=[StockCol.TICKER])

            df.columns = [str(c).strip().lower() for c in df.columns]

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()

        return df

    # ==========================================
    # 企業事件 (除權息 & 法說會) 資料庫操作
    # ==========================================
    def save_dividends_calendar(self, df: pd.DataFrame):
        """將除權息預告表寫入資料庫"""
        if df.empty: return
        with sqlite3.connect(self.db_path) as conn:
            # 實戰中只保留最新預告，直接 replace 最乾淨
            df.to_sql('dividends_calendar', conn, if_exists='replace', index=False)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_div_ticker ON dividends_calendar(ticker)')
            dbg.log("除權息行事曆已更新至資料庫。")

    def save_earnings_calendar(self, df: pd.DataFrame):
        """將法說會日程寫入資料庫"""
        if df.empty: return
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('earnings_calendar', conn, if_exists='replace', index=False)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_earn_ticker ON earnings_calendar(ticker)')
            dbg.log("法說會行事曆已更新至資料庫。")

    def get_upcoming_dividend(self, ticker: str, current_date: str) -> dict | None:
        """
        查詢該標的在指定日期「之後」最近一次的除權息資訊。
        回傳範例: {'ex_date': '2026-07-15', 'cash_dividend': 5.5}
        """
        query = f"""
            SELECT ex_date, cash_dividend
            FROM dividends_calendar
            WHERE ticker = ? AND ex_date >= ?
            ORDER BY ex_date ASC
            LIMIT 1
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(ticker, current_date))
            if not df.empty:
                return df.iloc[0].to_dict()
        return None

    def get_days_to_next_earnings(self, ticker: str, current_date: str) -> int | None:
        """
        計算距離下一次法說會還有幾天。
        若無即將到來的法說會，回傳 None。
        """
        query = f"""
            SELECT earnings_date
            FROM earnings_calendar
            WHERE ticker = ? AND earnings_date >= ?
            ORDER BY earnings_date ASC
            LIMIT 1
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(ticker, current_date))
            if not df.empty:
                earnings_date_str = df.iloc[0]['earnings_date']
                d1 = datetime.strptime(current_date, "%Y-%m-%d")
                d2 = datetime.strptime(earnings_date_str, "%Y-%m-%d")
                return (d2 - d1).days
        return None

    # 宏觀與籌碼指標 資料庫操作
    def save_macro_data(self, metric_name: str, df: pd.DataFrame):
        """
        將單一數值的宏觀指標存入資料庫 (例如: FUTURES_OI)
        預期傳入的 df 必須以 date 為 index，且只有一個數值欄位。
        """
        if df.empty: return

        # 假設 df 只有一個欄位，我們把它轉成 (metric_name, date, value) 的格式
        value_col = df.columns[0]
        records = []
        for date_idx, row in df.iterrows():
            date_str = str(date_idx).split(' ')[0]
            val = float(row[value_col]) if pd.notna(row[value_col]) else 0.0
            records.append((metric_name, date_str, val))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO macro_time_series (metric_name, date, value)
                VALUES (?, ?, ?)
            ''', records)
            conn.commit()
            dbg.log(f"成功儲存宏觀指標 [{metric_name}]，共 {len(records)} 筆。")

    def get_macro_data(self, metric_name: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """
        讀取指定的宏觀指標，回傳以日期為 Index 的 Pandas Series。
        """
        query = "SELECT date, value FROM macro_time_series WHERE metric_name = ?"
        params = [metric_name]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, index_col='date', parse_dates=['date'])

        if df.empty:
            return pd.Series(dtype=float, name=metric_name)

        # 整理 index 時區
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 回傳 Series，並將 name 設為該指標名稱
        series = df['value']
        series.name = metric_name.lower()
        return series
