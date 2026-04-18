import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.storages import JournalFileStorage, JournalStorage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from data.const import MacroTicker
from debug import dbg
from ml.const import FeatureCol
from ml.data.xgb_features import XGBFeatureEngine
from ml.engine import QuantAIEngine
from path import PathConfig

try:
    from ml.params import TrainConfig
    EARLY_STOP = TrainConfig.EARLY_STOP_ROUND
    N_SPLITS = TrainConfig.N_SPLITS
except ImportError:
    EARLY_STOP = 50
    N_SPLITS = 5

dbg.toggle()

# ==============================================================================
# 1. Optuna 核心目標函數
# ==============================================================================
def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': 1, # 限制 XGBoost 單核心運作，將多核算力交給外層的 Optuna 分配
        'n_estimators': 1000,
        'early_stopping_rounds': EARLY_STOP,

        # 配合嚴格 ATR 標籤，放寬一點深度，並加強雜訊過濾
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 1.0, 4.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    val_aucs = []

    for step, (train_index, val_index) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # 防禦機制：確保訓練集與驗證集都有兩種標籤
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            val_aucs.append(0.5)
            continue

        # 動態計算正負樣本權重 (因為現在是嚴格標籤，正樣本會變少)
        pos_count = max(sum(y_tr == 1), 1)
        neg_count = max(sum(y_tr == 0), 1)
        dynamic_scale_pos_weight = neg_count / pos_count

        model = xgb.XGBClassifier(**param, scale_pos_weight=dynamic_scale_pos_weight)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        val_aucs.append(auc)

        current_mean_auc = np.mean(val_aucs)
        trial.report(current_mean_auc, step)

        if step >= 2 and trial.should_prune():
            raise optuna.TrialPruned()

    if not val_aucs:
        return 0.0

    # 金融風險調整評分 (平滑化變異帶來的風險)
    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)
    penalty_factor = 0.5
    final_score = mean_auc - (penalty_factor * std_auc)

    trial.set_user_attr("mean_auc", float(mean_auc))
    trial.set_user_attr("std_auc", float(std_auc))

    return final_score


# ==============================================================================
# 2. 尋優主程式
# ==============================================================================
def run_optimization(target_total_trials: int, test_tickers: list, oos_days: int = 240):
    print("="*60)
    print("🚀 XGBoost 金融防過擬合尋優引擎啟動")
    print("="*60)

    print(f"⏳ 正在從資料庫萃取 {len(test_tickers)} 檔標的特徵資料...")
    df_list = []

    for ticker in test_tickers:
        try:
            engine = QuantAIEngine(ticker=ticker, oos_days=oos_days)
            macro_tickers = [e.value for e in MacroTicker]
            df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

            if df_raw.empty: continue

            df_train_raw = df_raw.iloc[:-oos_days] if oos_days > 0 else df_raw

            xgb_engine = XGBFeatureEngine()
            df_with_features = xgb_engine.process_pipeline(df_train_raw, lookahead=engine.config.lookahead, is_training=True)

            df_clean = df_with_features.dropna(subset=[FeatureCol.TARGET]).copy()
            if "ticker" not in df_clean.columns: df_clean["ticker"] = ticker
            df_list.append(df_clean)

        except Exception as e:
            print(f"⚠️ {ticker} 資料萃取失敗，略過: {e}")

    if not df_list:
        print("❌ 無法取得任何有效的訓練資料，程式中止。")
        return

    df_all = pd.concat(df_list, ignore_index=False)

    if "date" in df_all.columns:
        df_all = df_all.sort_values("date").reset_index(drop=True)
    elif isinstance(df_all.index, pd.DatetimeIndex) or df_all.index.name in ["Date", "date"]:
        df_all = df_all.sort_index().reset_index(drop=True)

    # 絕對白名單萃取，保證跟預測引擎特徵 100% 一致
    features = FeatureCol.get_features()

    # 檢查是否所有特徵都存在於 DataFrame 中
    missing_features = [f for f in features if f not in df_all.columns]
    if missing_features:
        print(f"❌ 嚴重錯誤：資料中缺少必須的特徵欄位: {missing_features}")
        return

    X_train = df_all[features]
    y_train = df_all[FeatureCol.TARGET].astype(int)

    print(f"✅ 全域資料合併完成！總訓練樣本數: {len(X_train)} 筆")
    print(f"🔍 實際訓練特徵數量: {len(features)} 欄")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
    log_path = PathConfig.RESULT_REPORT / "xgboost_optuna_study.journal.log"
    storage = JournalStorage(JournalFileStorage(str(log_path)))

    print(f"📁 XGBoost 尋優日誌連結至: {log_path.name}")

    study_name = "XGBoost_Robust_Model_v2_ATR"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    completed_trials = len(study.trials)
    remaining_trials = max(0, target_total_trials - completed_trials)

    if remaining_trials == 0:
        print(f"✅ 尋優專案已完成 {target_total_trials} 次測試，無需再跑！")
    else:
        print(f"⏳ 剩餘 {remaining_trials} 次測試即將開始...")
        n_jobs_to_use = -1

        with tqdm(total=remaining_trials, desc="🎯 XGBoost 尋優", unit="trial") as pbar:
            def update_tqdm_callback(study, trial):
                pbar.set_postfix({"Best Score": f"{study.best_value:.4f}"})
                pbar.update(1)

            study.optimize(
                lambda trial: objective(trial, X_train, y_train),
                n_trials=remaining_trials,
                callbacks=[update_tqdm_callback],
                n_jobs=n_jobs_to_use
            )

    print("\n\n" + "="*60)
    print("🏆 【尋優完成】最強 XGBoost 參數誕生！")
    print("="*60)

    if len(study.trials) > 0 and study.best_trial:
        print(f"🥇 最高風險調整 AUC 分數: {study.best_value:.4f}")
        print("\n📝 請將以下參數寫入您的 XGBHyperParams (位於 src/ml/params.py)：")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    else:
        print("⚠️ 尚未產生任何有效的尋優結果。")
    print("="*60)

if __name__ == "__main__":
    test_tickers = [
        "0052.TW", "2324.TW","3006.TW", "2301.TW",
        "3481.TW", "9958.TW", "2344.TW",
        "2382.TW", "2377.TW", "2454.TW", "1519.TW" "2337.TW",
    ]

    # 強烈建議：對於 XGBoost 來說 300 次已經逼近全域最佳解了
    target_total_trials = 1200

    run_optimization(target_total_trials=target_total_trials, test_tickers=test_tickers)