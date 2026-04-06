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
        """
        df_stock = self.get_daily_data(stock_ticker)
        if df_stock.empty:
            return df_stock

        aligned_df = df_stock.copy()
        overseas_tickers = MacroTicker.get_overseas_tickers()
        macro_cols = [] # 記錄所有加入的大盤欄位

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

        if macro_cols:
            aligned_df[macro_cols] = aligned_df[macro_cols].ffill()

        return aligned_df

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