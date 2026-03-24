import pandas as pd
from data.fetcher import Fetcher
from data.manager import DataManager
from data.variable import TimeUnit
from debug import dbg
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer

if __name__ == "__main__":
    ticker = "0052.TW"
    db = DataManager()
    fetcher = Fetcher() # 假設你已經定義好這個類別

    # --- 第一步：抓取並存儲 (確保教科書已經準備好) ---
    dbg.log(f"正在更新 {ticker} 的歷史資料...")
    # 抓取最近 10 年的資料來進行訓練，訓練 AI 需要夠長的歷史
    daily_df = fetcher.fetch_daily_data(ticker, period=5, unit=TimeUnit.YEAR)
    if not daily_df.empty:
        db.save_daily_data(ticker, daily_df)
    else:
        dbg.error("抓取資料失敗，請檢查網路或 Ticker 名稱。")
        exit()

    # --- 第二步：從 DB 讀取資料 (現在保證裡面有東西了) ---
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        df_raw = pd.read_sql(
            f"SELECT * FROM daily_k_lines WHERE ticker='{ticker}' ORDER BY date",
            conn, index_col='date', parse_dates=['date']
        )

    # --- 第三步：特徵工程與訓練 ---
    if not df_raw.empty:
        # 特徵工程 (波段模式通常使用日 K)
        engine = XGBFeatureEngine()
        df_clean = engine.process_pipeline(df_raw, lookahead=20)

        # 訓練與驗證
        trainer = XGBTrainer()
        trainer.train_with_cv(df_clean, n_splits=5)

        # 存檔供 UI 使用
        trainer.train_and_save_final_model(df_clean)
