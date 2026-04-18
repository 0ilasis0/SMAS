import gc
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score
from tqdm import tqdm

from data.const import MacroTicker
from debug import dbg
from ml.const import DLModelType, RNNType
from ml.data.dl_features import DLFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import DLHyperParams
from ml.trainers.dl_trainer import DLTrainer
from path import PathConfig

dbg.toggle()

# 強制固定所有隨機種子，確保比較實驗的公平性
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 若有使用 MacOS MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def run_dl_comparison(test_tickers: list, oos_days: int = 240):
    print("="*70)
    print("🧠 深度學習 (DL) 樣本外盲測 (OOS) 效能對照引擎")
    print("="*70)

    BASELINE_PARAMS = {
        'CNN_OUT_CHANNELS': 16,
        'LSTM_HIDDEN': 32,
        'LEARNING_RATE': 0.002,
        'DROPOUT': 0.2,
        'BATCH_SIZE': 32,
        'EPOCHS': 50
    }

    # 請替換為您的 Optuna 結果
    OPTIMIZED_PARAMS = {
        'CNN_OUT_CHANNELS': 32,
        'LSTM_HIDDEN': 64,
        'LEARNING_RATE': 0.0015,
        'DROPOUT': 0.35,
        'BATCH_SIZE': 32,
        'EPOCHS': 50
    }

    def apply_dl_params(params):
        DLHyperParams.CNN_OUT_CHANNELS = params['CNN_OUT_CHANNELS']
        DLHyperParams.LSTM_HIDDEN = params['LSTM_HIDDEN']
        DLHyperParams.LEARNING_RATE = params['LEARNING_RATE']
        DLHyperParams.DROPOUT = params['DROPOUT']
        DLHyperParams.BATCH_SIZE = params['BATCH_SIZE']
        DLHyperParams.EPOCHS = params['EPOCHS']

    report_data = []
    print(f"⏳ 開始針對 {len(test_tickers)} 檔個股進行 DL 訓練與盲測比較...\n")

    for ticker in tqdm(test_tickers, desc="評估進度"):
        try:
            # 每次訓練一檔新股票前，重置種子，保證對照組與實驗組起點相同
            seed_everything(42)

            engine = QuantAIEngine(ticker=ticker, oos_days=oos_days,
                                   dl_model_type=DLModelType.HYBRID, rnn_type=RNNType.LSTM)
            macro_tickers = MacroTicker.get_all_tickers()

            # 這裡拿到的 df_raw 包含了訓練 + OOS 的所有歷史資料
            df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

            if df_raw.empty or len(df_raw) < (oos_days + 100):
                continue

            # 防堵 Scaler 洩漏！先切 DataFrame，再送進特徵引擎
            df_train_raw = df_raw.iloc[:-oos_days]

            # 加入熱機緩衝 (Warm-up Buffer)
            # 多切 60 天的資料給 OOS，讓均線跟 LSTM 的 TIME_STEPS 有歷史資料可以算
            buffer_days = 60
            df_oos_raw_with_buffer = df_raw.iloc[-(oos_days + buffer_days):]

            # 訓練集處理 (Scaler 會 fit 訓練集)
            dl_engine_train = DLFeatureEngine(engine.config.lookahead)
            X_train, y_train, _ = dl_engine_train.process_pipeline(df_train_raw, is_training=True)

            # OOS 處理 (使用帶有緩衝區的資料)
            X_oos_full, y_oos_full, _ = dl_engine_train.process_pipeline(df_oos_raw_with_buffer, is_training=False)

            if X_train is None or X_oos_full is None:
                continue

            # 精準切掉熱機緩衝，只保留真正的 OOS 天數
            # 確保我們只評估模型在真正盲測期間的表現
            X_oos = X_oos_full[-oos_days:]
            y_oos = y_oos_full[-oos_days:]

            if len(np.unique(y_oos)) < 2:
                continue

            # ==========================================
            # A. 測試對照組 (Baseline)
            # ==========================================
            seed_everything(42) # 確保 Baseline 模型初始化狀態
            apply_dl_params(BASELINE_PARAMS)
            trainer_base = DLTrainer(ticker, DLModelType.HYBRID, RNNType.LSTM)

            # 假設您的 trainer 有支援驗證集早停，請不要傳入 X_oos！這裡必須盲測到底
            trainer_base.train(X_train, y_train)
            preds_base = trainer_base.predict(X_oos)

            pr_auc_base = average_precision_score(y_oos, preds_base)
            threshold_base = np.percentile(preds_base, 80)
            labels_base = (preds_base >= threshold_base).astype(int)
            prec_base = precision_score(y_oos, labels_base, zero_division=0)

            # ==========================================
            # B. 測試實驗組 (Optimized)
            # ==========================================
            seed_everything(42) # 確保 Optimized 模型初始化狀態相同
            apply_dl_params(OPTIMIZED_PARAMS)
            trainer_opt = DLTrainer(ticker, DLModelType.HYBRID, RNNType.LSTM)

            trainer_opt.train(X_train, y_train)
            preds_opt = trainer_opt.predict(X_oos)

            pr_auc_opt = average_precision_score(y_oos, preds_opt)
            threshold_opt = np.percentile(preds_opt, 80)
            labels_opt = (preds_opt >= threshold_opt).astype(int)
            prec_opt = precision_score(y_oos, labels_opt, zero_division=0)

            # ==========================================
            # C. 寫入報表
            # ==========================================
            report_data.append({
                "Ticker": ticker,
                "PR_AUC_Before": round(pr_auc_base, 4),
                "PR_AUC_After": round(pr_auc_opt, 4),
                "PR_AUC_Diff": round(pr_auc_opt - pr_auc_base, 4),
                "Top20_Prec_Before(%)": round(prec_base * 100, 2),
                "Top20_Prec_After(%)": round(prec_opt * 100, 2),
                "Top20_Prec_Diff(%)": round((prec_opt - prec_base) * 100, 2)
            })

            # 強制記憶體回收，避免連續訓練導致 GPU 崩潰
            del trainer_base, trainer_opt, X_train, y_train, X_oos, y_oos
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"\n⚠️ {ticker} 評估失敗: {e}")

    # ================= 產出 CSV 與總結 =================
    if not report_data:
        print("❌ 無法產生任何報表數據。")
        return

    df_report = pd.DataFrame(report_data)
    df_report = df_report.sort_values(by="PR_AUC_Diff", ascending=False)

    PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
    csv_path = PathConfig.RESULT_REPORT / "dl_model_comparison_report.csv"
    df_report.to_csv(csv_path, index=False)

    print("\n" + "="*70)
    print("🏆 深度學習 (DL) 盲測績效總結")
    print("="*70)
    avg_prauc_before = df_report["PR_AUC_Before"].mean()
    avg_prauc_after = df_report["PR_AUC_After"].mean()
    avg_prec_before = df_report["Top20_Prec_Before(%)"].mean()
    avg_prec_after = df_report["Top20_Prec_After(%)"].mean()

    print(f"📉 調整前 平均 PR-AUC: \t{avg_prauc_before:.4f}")
    print(f"🚀 調整後 平均 PR-AUC: \t{avg_prauc_after:.4f} (提升: {avg_prauc_after - avg_prauc_before:+.4f})")
    print("-" * 70)
    print(f"📉 調整前 Top 20% 抓漲精準度: {avg_prec_before:.2f}%")
    print(f"🚀 調整後 Top 20% 抓漲精準度: {avg_prec_after:.2f}% (提升: {avg_prec_after - avg_prec_before:+.2f}%)")
    print("="*70)

if __name__ == "__main__":
    test_tickers = [
        "0050.TW", "0052.TW", "2330.TW", "2317.TW", "2454.TW",
        "2382.TW", "2377.TW", "3231.TW", "2324.TW", "2301.TW",
        "2603.TW", "2881.TW", "2409.TW", "3481.TW", "2344.TW",
        "2455.TW", "2388.TW", "1519.TW"
    ]

    run_dl_comparison(test_tickers=test_tickers)