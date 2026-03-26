import pandas as pd

from data.fetcher import Fetcher
from data.manager import DataManager
from data.variable import TimeUnit
from debug import dbg
from ml.dl_features import DLFeatureEngine
from ml.dl_trainer import DLTrainer
from ml.meta_learner import MetaLearner
from ml.params import FeatureCol, RNNType
from ml.xgb_features import XGBFeatureEngine
from ml.xgb_trainer import XGBTrainer

if __name__ == "__main__":
    ticker = "2388.TW"
    fetcher = Fetcher()
    db = DataManager()

    # 抓取並存儲歷史資料 ---
    # dbg.log(f"正在更新 {ticker} 的歷史資料...")
    # daily_df = fetcher.fetch_daily_data(ticker)
    # if not daily_df.empty:
    #     db.save_daily_data(ticker, daily_df)
    # else:
    #     dbg.error("抓取資料失敗，請檢查網路或 Ticker 名稱。")
    #     exit()


    # ==========================================
    # 0. 取得資料
    # ==========================================
    df_raw = db.get_daily_data(ticker)

    if df_raw.empty:
        dbg.error(f"找不到 {ticker} 的資料，請先執行抓取。")
        exit()

    # ==========================================
    # Level 0 - 左腦：XGBoost 處理管線
    # ==========================================
    xgb_engine = XGBFeatureEngine()
    df_xgb_clean = xgb_engine.process_pipeline(df_raw)

    xgb_trainer = XGBTrainer(ticker)
    # 取得 XGBoost 的 OOF 預測
    oof_xgb = xgb_trainer.train_with_cv(df_xgb_clean)

    # 取得真實標籤 (用 XGBoost 的 DataFrame 當作對齊基準)
    y_true = df_xgb_clean[FeatureCol.TARGET]

    # ==========================================
    # Level 0 - 右腦：DL (CNN-GRU/LSTM) 處理管線
    # ==========================================
    df_with_indicators = xgb_engine._create_daily_features(df_raw)

    dl_engine = DLFeatureEngine()
    X_dl, y_dl, _, original_index = dl_engine.process_pipeline(df_with_indicators)
    dl_trainer = DLTrainer(ticker=ticker, rnn_type=RNNType.GRU)

    # 取得 DL 的 OOF 預測
    oof_dl = dl_trainer.train_with_cv(X_dl, y_dl, original_index)

    # ==========================================
    # Level 1 - 總指揮：Meta-Learner 縫合雙腦
    # ==========================================
    # 這裡更新了 ticker 參數的傳入
    meta_learner = MetaLearner(ticker=ticker)
    meta_learner.train_and_evaluate(oof_xgb, oof_dl, y_true)

    dbg.log("\n🎉 Stacking 整合訓練管線全數執行完畢！")
