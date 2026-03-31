import sqlite3
from pathlib import Path

import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
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

            # 日 K 線表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_k_lines (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            ''')

            # 分時 K 線表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intraday_k_lines (
                    ticker TEXT,
                    datetime TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, datetime)
                )
            ''')

            # 自選股清單
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_watchlist (
                    ticker TEXT PRIMARY KEY,
                    added_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def clear_ticker_data(self, ticker: str):
        """
        刪除特定標的的所有歷史日線與分時資料。
        用於確保寫入新抓取的「還原股價」時，不會與舊的未還原髒資料發生混層。
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM daily_k_lines WHERE ticker = ?", (ticker,))
            cursor.execute("DELETE FROM intraday_k_lines WHERE ticker = ?", (ticker,))
            conn.commit()
            dbg.log(f"已徹底清空 {ticker} 的歷史資料快取。")

    def save_daily_data(self, ticker: str, df: pd.DataFrame):
        """將日 K 線 DataFrame 存入 SQLite"""
        if df.empty: return

        df = df.copy()
        df = df.dropna(subset=[StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.CLOSE], ignore_index=False)

        if df.empty:
            dbg.war(f"[{ticker}] 資料清洗後為空，取消寫入。")
            return

        df.columns = [str(c).strip().capitalize() for c in df.columns]

        records = [
            (
                ticker,
                row.Index.strftime('%Y-%m-%d') if hasattr(row.Index, 'strftime') else str(row.Index),
                row.Open, row.High, row.Low, row.Close,
                int(row.Volume) if pd.notna(row.Volume) else 0
            )
            for row in df.itertuples()
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO daily_k_lines
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            dbg.log(f"成功儲存 {ticker} 日 K 線資料，共 {len(records)} 筆。")

    def save_intraday_data(self, ticker: str, df: pd.DataFrame):
        """將分時 K 線 DataFrame 存入 SQLite"""
        if df.empty: return

        df = df.copy()
        df.columns = [str(c).strip().capitalize() for c in df.columns]

        records = [
            (
                ticker,
                row.Index.strftime('%Y-%m-%d %H:%M:%S') if hasattr(row.Index, 'strftime') else str(row.Index),
                row.Open, row.High, row.Low, row.Close,
                int(row.Volume) if pd.notna(row.Volume) else 0
            )
            for row in df.itertuples()
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO intraday_k_lines
                (ticker, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
            dbg.log(f"成功儲存 {ticker} 分時 K 線資料，共 {len(records)} 筆。")

    def add_to_watchlist(self, ticker: str):
        """新增標的至自選股清單"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO user_watchlist (ticker) VALUES (?)
            ''', (ticker,))
            conn.commit()
            dbg.log(f"已將 {ticker} 加入自選清單。")

    def get_daily_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        return self._fetch_data(
            table_name='daily_k_lines', time_col='date', ticker=ticker, start_time=start_date, end_time=end_date
        )

    def get_intraday_data(self, ticker: str, start_datetime: str = None, end_datetime: str = None) -> pd.DataFrame:
        return self._fetch_data(
            table_name='intraday_k_lines', time_col='datetime', ticker=ticker, start_time=start_datetime, end_time=end_datetime
        )

    def get_aligned_market_data(self, stock_ticker: str, macro_tickers: list[str]) -> pd.DataFrame:
        """
        機構級數據對齊引擎
        以個股交易日為主體 (Left Join)，將大盤數據併入，並自動處理美股時差與休市問題。
        """
        df_stock = self.get_daily_data(stock_ticker)
        if df_stock.empty:
            return df_stock

        aligned_df = df_stock.copy()
        overseas_tickers = MacroTicker.get_overseas_tickers()

        for mt in macro_tickers:
            df_macro = self.get_daily_data(mt)
            if df_macro.empty: continue

            # 加上大盤字首
            prefix = mt.replace('^', '') + "_"
            df_macro = df_macro.add_prefix(prefix)

            if mt in overseas_tickers:
                # 1. 將美股本身往前補值 (處理美國國定假日，確保每天都有最後收盤價)
                # 注意：這裡不能用全宇宙日期，直接用它自己的 index 填補即可，否則會跑出未來數據
                df_macro_ffilled = df_macro.asfreq('D', method='ffill')

                # 2. 將補值後的美股資料往未來推遲一天 (模擬台灣早上看到昨晚美股收盤 T+1)
                df_macro_shifted = df_macro_ffilled.shift(1)

                # 3. 把推遲後的美股貼近台股主體 (只保留台股有開盤的日期)
                aligned_df = aligned_df.join(df_macro_shifted, how='left')
            else:
                # 國內大盤 (如 TWII) 無時差，直接 Left Join
                aligned_df = aligned_df.join(df_macro, how='left')

        # 最前端因 shift(1) 產生的 NaN 會保留，交由後續特徵工程處理
        # 這裡再補一次 ffill，確保台股遇到颱風假之類造成的 NaN 被補齊
        aligned_df = aligned_df.ffill()
        return aligned_df

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
            df = pd.read_sql_query(query, conn, params=params, index_col=time_col, parse_dates=[time_col])

        if not df.empty:
            if StockCol.TICKER in df.columns:
                df = df.drop(columns=[StockCol.TICKER])

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.index = pd.to_datetime(df.index)

            df = df.sort_index()

        return df
