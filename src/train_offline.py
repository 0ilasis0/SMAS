from data.manager import DataManager
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer

if __name__ == "__main__":
    ticker = "0052.TW"

    # 1. 從本地資料庫讀取資料 (這裡假設你資料庫已經有抓好的 daily_k_lines)
    db = DataManager()
    # 我們需要寫一個簡單的方法從 SQLite 把資料撈成 DataFrame
    # 這裡假設你在 DataManager 加了 load_daily_data(ticker) 的方法
    import sqlite3

    import pandas as pd
    with sqlite3.connect(db.db_path) as conn:
        df_raw = pd.read_sql(f"SELECT * FROM daily_k_lines WHERE ticker='{ticker}' ORDER BY date", conn, index_col='date', parse_dates=['date'])

    if not df_raw.empty:
        # 2. 執行特徵工程
        engine = XGBFeatureEngine()
        df_clean = engine.process_pipeline(df_raw, lookahead=20)

        # 3. 訓練模型
        trainer = XGBTrainer()
        # 看看歷史交叉驗證表現
        trainer.train_with_cv(df_clean, n_splits=5)
        # 儲存最終模型供實戰使用
        trainer.train_and_save_final_model(df_clean)