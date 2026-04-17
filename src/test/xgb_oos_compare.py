import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from tqdm import tqdm

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.data.xgb_features import XGBFeatureEngine
from ml.engine import QuantAIEngine
# ================= 引入您的實際生產模組 =================
from path import PathConfig

dbg.toggle()

def run_xgb_comparison(test_tickers: list, lookahead: int, oos_days: int = 240):
    print("="*70)
    print("🕵️‍♂️ XGBoost 樣本外盲測 (OOS) 效能對照引擎啟動")
    print("="*70)

    # ================= 定義對照組與實驗組參數 =================
    # 🔴 對照組：原本的 XGBHyperParams 預設值
    BASELINE_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': 1,
        'n_estimators': 100,

        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # 🟢 實驗組：請將 Optuna 尋優跑出來的最強參數貼在這邊
    OPTIMIZED_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': 1,
        'n_estimators': 100,

        # --- ⬇️ 請替換為您的 Optuna 結果 ⬇️ ---
        'max_depth': 3,
        'min_child_weight': 11,
        'learning_rate': 0.0992,
        'subsample': 0.6505,
        'colsample_bytree': 0.5393,
        'gamma': 3.4522,
        'reg_alpha': 2.2545,
        'reg_lambda': 0.0779
    }

    report_data = []

    print(f"⏳ 開始針對 {len(test_tickers)} 檔個股進行訓練與盲測比較...\n")

    # ================= 3. 逐檔股票進行回測與比較 =================
    for ticker in tqdm(test_tickers, desc="評估進度"):
        try:
            engine = QuantAIEngine(ticker=ticker, oos_days=oos_days)
            macro_tickers = [e.value for e in MacroTicker]
            df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

            if df_raw.empty: continue

            xgb_engine = XGBFeatureEngine()
            df_with_features = xgb_engine.process_pipeline(df_raw, lookahead=lookahead, is_training=True)
            df_clean = df_with_features.dropna(subset=[FeatureCol.TARGET]).copy()

            # 訓練集只取到倒數 (oos_days + lookahead) 天
            df_train_full = df_clean.iloc[: -(oos_days + lookahead)]
            df_oos = df_clean.iloc[-oos_days:]

            exclude_cols = [FeatureCol.TARGET, "date", "ticker", "Date", StockCol.CLOSE.value, StockCol.VOLUME.value, StockCol.OPEN.value, StockCol.HIGH.value, StockCol.LOW.value]
            features = [c for c in df_train_full.columns if c not in exclude_cols]

            X_train_full, y_train_full = df_train_full[features], df_train_full[FeatureCol.TARGET].astype(int)
            X_oos, y_oos = df_oos[features], df_oos[FeatureCol.TARGET].astype(int)

            if len(np.unique(y_oos)) < 2:
                continue

            pos_count = max(sum(y_train_full == 1), 1)
            neg_count = max(sum(y_train_full == 0), 1)
            scale_pos_weight = neg_count / pos_count

            # --- B. 訓練與評估：Baseline (調整前) ---
            model_base = xgb.XGBClassifier(**BASELINE_PARAMS, scale_pos_weight=scale_pos_weight)
            model_base.fit(X_train_full, y_train_full, verbose=False)

            preds_proba_base = model_base.predict_proba(X_oos)[:, 1]
            preds_label_base = (preds_proba_base >= 0.5).astype(int)

            auc_base = roc_auc_score(y_oos, preds_proba_base)
            acc_base = accuracy_score(y_oos, preds_label_base)
            prec_base = precision_score(y_oos, preds_label_base, zero_division=0)

            # --- C. 訓練與評估：Optimized (調整後) ---

            # 從訓練集內部再切出最後 20% 當作 Valid Set 供 Early Stopping 使用
            val_split_idx = int(len(X_train_full) * 0.8)
            X_tr, y_tr = X_train_full.iloc[:val_split_idx], y_train_full.iloc[:val_split_idx]
            X_val, y_val = X_train_full.iloc[val_split_idx:], y_train_full.iloc[val_split_idx:]

            # 直接使用含有 early_stopping_rounds 與 n_estimators=1000 的 OPTIMIZED_PARAMS
            model_opt = xgb.XGBClassifier(**OPTIMIZED_PARAMS, scale_pos_weight=scale_pos_weight)

            # 使用訓練集內部的 X_val 決定最佳樹量
            model_opt.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # 以最佳樹量對盲測集 (X_oos) 進行預測
            preds_proba_opt = model_opt.predict_proba(X_oos)[:, 1]
            preds_label_opt = (preds_proba_opt >= 0.5).astype(int)

            auc_opt = roc_auc_score(y_oos, preds_proba_opt)
            acc_opt = accuracy_score(y_oos, preds_label_opt)
            prec_opt = precision_score(y_oos, preds_label_opt, zero_division=0)

            # --- D. 寫入報表 (已修正重複寫入問題) ---
            report_data.append({
                "Ticker": ticker,
                "AUC_Before": round(auc_base, 4),
                "AUC_After": round(auc_opt, 4),
                "AUC_Diff": round(auc_opt - auc_base, 4),
                "Acc_Before(%)": round(acc_base * 100, 2),
                "Acc_After(%)": round(acc_opt * 100, 2),
                "Precision_Before(%)": round(prec_base * 100, 2),
                "Precision_After(%)": round(prec_opt * 100, 2),
                "Opt_Trees": model_opt.best_iteration if hasattr(model_opt, 'best_iteration') else "N/A"
            })

        except Exception as e:
            print(f"\n⚠️ {ticker} 評估失敗: {e}")

    # ================= 4. 產出 CSV 比較報告 =================
    if not report_data:
        print("❌ 無法產生任何報表數據。")
        return

    df_report = pd.DataFrame(report_data)

    # 依照 AUC 進步幅度排序
    df_report = df_report.sort_values(by="AUC_Diff", ascending=False)

    PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
    csv_path = PathConfig.RESULT_REPORT / "xgb_model_comparison_report.csv"
    df_report.to_csv(csv_path, index=False)

    # ================= 印出終極總結 =================
    print("\n" + "="*70)
    print("🏆 XGBoost 盲測績效總結")
    print("="*70)

    avg_auc_before = df_report["AUC_Before"].mean()
    avg_auc_after = df_report["AUC_After"].mean()
    avg_prec_before = df_report["Precision_Before(%)"].mean()
    avg_prec_after = df_report["Precision_After(%)"].mean()

    print(f"📉 調整前 (Baseline) 平均 AUC: \t{avg_auc_before:.4f}")
    print(f"🚀 調整後 (Optuna)   平均 AUC: \t{avg_auc_after:.4f} (提升: {avg_auc_after - avg_auc_before:+.4f})")
    print("-" * 70)
    print(f"📉 調整前 (Baseline) 平均抓漲精準度: {avg_prec_before:.2f}%")
    print(f"🚀 調整後 (Optuna)   平均抓漲精準度: {avg_prec_after:.2f}% (提升: {avg_prec_after - avg_prec_before:+.2f}%)")
    print("="*70)
    print(f"📁 詳細個股對照 CSV 已儲存至: {csv_path}")

if __name__ == "__main__":
    from ml.params import SessionConfig
    lookahead = SessionConfig.lookahead

    test_tickers = [
        "2324.TW",
        "3481.TW", "0052.TW", "2481.TW",
        "2344.TW", "4919.TW", "3231.TW", "2455.TW", "9958.TW",
        "3006.TW", "2301.TW", "4916.TW"
    ]

    run_xgb_comparison(test_tickers=test_tickers, lookahead=lookahead)