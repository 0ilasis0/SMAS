import sqlite3
from pathlib import Path

import pandas as pd

from data.const import StockCol
from debug import dbg
from path import PathConfig


class DataManager:
    def __init__(self, db_path: str = PathConfig.IDSS_DATA):
        self.db_path = db_path

        self.setup()

    def setup(self):
        db_path_obj = Path(self.db_path)
        db_dir = db_path_obj.parent

        # 若目錄不存在，自動建立
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            dbg.log(f"已自動建立資料庫目錄: {db_dir}")

        self._create_tables()

    def _create_tables(self):
        """初始化資料表 (Table Schema)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 日 K 線表 (使用 ticker 與 date 作為複合主鍵)
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

            # 分時 K 線表 (使用 ticker 與 datetime 作為複合主鍵)
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

    def save_daily_data(self, ticker: str, df: pd.DataFrame):
        """將日 K 線 DataFrame 存入 SQLite"""
        if df.empty:
            return

        records = []
        for index, row in df.iterrows():
            # pandas 的 index (Date) 轉為 YYYY-MM-DD 字串
            date_str = index.strftime('%Y-%m-%d')
            records.append((
                ticker, date_str,
                row[StockCol.OPEN],
                row[StockCol.HIGH],
                row[StockCol.LOW],
                row[StockCol.CLOSE],
                int(row[StockCol.VOLUME])
            ))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # INSERT OR REPLACE：確保同一檔股票在同一天不會有重複資料
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

        records = []
        for index, row in df.iterrows():
            # 轉換為包含時間的字串 YYYY-MM-DD HH:MM:SS
            datetime_str = index.strftime('%Y-%m-%d %H:%M:%S')
            records.append((
                ticker, datetime_str,
                row[StockCol.OPEN],
                row[StockCol.HIGH],
                row[StockCol.LOW],
                row[StockCol.CLOSE],
                int(row[StockCol.VOLUME])
            ))

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
        """
        從資料庫讀取指定標的的日 K 線資料
        可選填 start_date 與 end_date (格式: 'YYYY-MM-DD') 來過濾區間。
        """
        return self._fetch_data(
            table_name='daily_k_lines',
            time_col='date',
            ticker=ticker,
            start_time=start_date,
            end_time=end_date
        )

    def get_intraday_data(self, ticker: str, start_datetime: str = None, end_datetime: str = None) -> pd.DataFrame:
        """從資料庫讀取指定標的的分時 K 線資料"""
        return self._fetch_data(
            table_name='intraday_k_lines',
            time_col='datetime',
            ticker=ticker,
            start_time=start_datetime,
            end_time=end_datetime
        )

    def _fetch_data(self, table_name: str, time_col: str, ticker: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        """通用的資料庫查詢與 DataFrame 轉換邏輯"""
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

        # 整理 DataFrame：剔除不參與訓練的 ticker 欄位
        if not df.empty and 'ticker' in df.columns:
            df = df.drop(columns=['ticker'])

        return df
