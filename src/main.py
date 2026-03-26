import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from data.fetcher import Fetcher  # 新增：匯入 Fetcher
from data.manager import DataManager
from data.variable import TimeUnit  # 新增：匯入 TimeUnit
from debug import dbg
from ml.dl_features import DLFeatureEngine
from ml.dl_trainer import DLTrainer
from ml.params import RNNType
from ml.xgb_features import XGBFeatureEngine

if __name__ == "__main__":
    # 設定測試標的
    ticker = "0052.TW"

    db = DataManager()
    fetcher = Fetcher()

    # --- 第一步：抓取並存儲歷史資料 ---
    # dbg.log(f"正在更新 {ticker} 的歷史資料...")
    # daily_df = fetcher.fetch_daily_data(ticker, period=10, unit=TimeUnit.YEAR)
    # if not daily_df.empty:
    #     db.save_daily_data(ticker, daily_df)
    # else:
    #     dbg.error("抓取資料失敗，請檢查網路或 Ticker 名稱。")
    #     exit()

    # --- 第二步：從 DB 讀取資料 ---
    # 使用你剛寫好的 API，優雅地取得資料
    df_raw = db.get_daily_data(ticker)

    if df_raw.empty:
        dbg.error(f"找不到 {ticker} 的資料，請檢查資料庫。")
        exit()

    # --- 第三步：特徵工程 ---
    # 先計算共用的技術指標 (RSI, Bias, MACD 等)
    indicator_engine = XGBFeatureEngine()
    df_with_indicators = indicator_engine._create_daily_features(df_raw)
    df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan)

    # 再將帶有指標的資料送給 DL 引擎進行「切片」與「標準化」
    dl_engine = DLFeatureEngine()
    X, y, _ = dl_engine.process_pipeline(df_with_indicators)

    # 取得對應的日期標籤 (用於 Series Index)
    original_index = df_raw.index[-len(y):]

    # --- 第四步：自動化 A/B 測試 (CV 模擬考) ---
    performance_summary = []

    for rnn_type in [RNNType.LSTM, RNNType.GRU]:
        dbg.log(f"\n{'='*20} 開始測試模型: CNN-{rnn_type.name} {'='*20}")

        trainer = DLTrainer(ticker=ticker, rnn_type=rnn_type)

        # 執行 CV 並收集 OOF 預測 (Out-of-Fold)
        oof_preds = trainer.train_with_cv(X, y, original_index)

        # 計算該模型在全量 OOF 上的最終表現
        if not oof_preds.empty:
            y_pred_binary = (oof_preds.values > 0.5).astype(int)
            y_true = y[-len(oof_preds):]

            acc = accuracy_score(y_true, y_pred_binary)
            auc = roc_auc_score(y_true, oof_preds.values)

            performance_summary.append({
                "Model": f"CNN-{rnn_type.name}",
                "Accuracy": acc,
                "AUC": auc
            })

            dbg.log(f"[{rnn_type.name} 總體結果] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

        # 清除 GPU/MPS 快取，避免記憶體爆炸
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        dbg.log(f"{'='*20} 模型: CNN-{rnn_type.name} 測試結束 {'='*20}\n")

    # --- 第五步：顯示勝負表並訓練最終模型 (正式考試與產出 pth) ---
    dbg.log("\n" + "#"*30)
    dbg.log("🏆 A/B 測試最終勝負表 🏆")
    summary_df = pd.DataFrame(performance_summary)
    print(summary_df.to_string(index=False))

    if len(performance_summary) > 1:
        # 自動找出 AUC 最高的模型
        winner = summary_df.loc[summary_df['AUC'].idxmax()]
        dbg.log(f"\n➔ 根據 AUC 表現，自動選定冠軍: {winner['Model']}")

        # 🚀 這裡就是產生 .pth 檔案的關鍵點！
        dbg.log(f"準備使用全量資料為冠軍模型 ({winner['Model']}) 進行最終訓練...")
        final_rnn = RNNType.LSTM if "LSTM" in winner['Model'] else RNNType.GRU

        final_trainer = DLTrainer(ticker=ticker, rnn_type=final_rnn)
        # 這行跑完，你的 data/processed/model 資料夾就會出現 pth 檔案了
        final_trainer.train_and_save_final_model(X, y)

        dbg.log("🎉 訓練與存檔流程全數完畢！")