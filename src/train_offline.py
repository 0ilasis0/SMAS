import pandas as pd
from data.fetcher import Fetcher
from data.manager import DataManager
from data.variable import TimeUnit
from debug import dbg
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer

from path import PathConfig

if __name__ == "__main__":
    ticker = "2388.TW"
    db = DataManager()
    fetcher = Fetcher()

    # --- 第一步：抓取並存儲 ---
    dbg.log(f"正在更新 {ticker} 的歷史資料...")
    daily_df = fetcher.fetch_daily_data(ticker, period=10, unit=TimeUnit.YEAR)
    if not daily_df.empty:
        db.save_daily_data(ticker, daily_df)
    else:
        dbg.error("抓取資料失敗，請檢查網路或 Ticker 名稱。")
        exit()

    # --- 第二步：從 DB 讀取資料 ---
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        df_raw = pd.read_sql(
            f"SELECT * FROM daily_k_lines WHERE ticker='{ticker}' ORDER BY date",
            conn, index_col='date', parse_dates=['date']
        )

    # --- 第三步：特徵工程與訓練 ---
    if not df_raw.empty:
        # 特徵工程
        engine = XGBFeatureEngine()
        df_clean = engine.process_pipeline(df_raw, lookahead=20)

        # 訓練與驗證，並用變數接住 OOF 預測結果
        trainer = XGBTrainer()
        oof_preds = trainer.train_with_cv(df_clean, n_splits=5)

        # 將預測結果與真實股價結合，進行初步分析
        if not oof_preds.empty:
            # 將機率序列新增為 df_clean 的一個新欄位
            df_clean['predict_proba'] = oof_preds

            # 找出那些 AI 極度有信心會漲的日子 (機率 > 0.7)
            strong_buy_signals = df_clean[df_clean['predict_proba'] > 0.7]

            dbg.log(f"回測分析：在歷史資料中，AI 共發出了 {len(strong_buy_signals)} 次強烈看漲訊號。")

            # 🚀 動態生成 CSV 路徑
            save_path = PathConfig.get_backtest_report_path(ticker)

            # 存檔，並加入 utf-8-sig 解決 Windows Excel 亂碼問題
            df_clean.to_csv(save_path, encoding='utf-8-sig')
            dbg.log(f"✅ 回測報告已成功儲存至: {save_path}")

        # 存檔供 UI 使用
        trainer.train_and_save_final_model(df_clean)
