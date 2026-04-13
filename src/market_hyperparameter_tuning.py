import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.storages import JournalFileStorage, JournalStorage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from data.const import MacroTicker
from debug import dbg
from ml.const import MarketFeatureCol
from ml.data.market_features import MarketFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import TrainConfig
from path import PathConfig

dbg.toggle()

def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series):
    """大盤防禦專屬目標函數 (極簡主義 + 高度正則化)"""

    # 🌟 核心：嚴格限制樹的複雜度
    max_depth = trial.suggest_int('max_depth', 2, 5) # 大盤資料少，最多只能切 2~5 刀

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': 1, # 讓 Optuna 管多核
        'verbose': -1,
        'n_estimators': 1000,

        'max_depth': max_depth,
        # 限制葉子節點數，絕對不能超過 2^max_depth
        'num_leaves': trial.suggest_int('num_leaves', 3, int(2**max_depth) - 1),

        # 🌟 要求每個葉子要有足夠天數，強迫抓大波段，不背單日特例
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),

        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),

        # 🌟 L1/L2 正則化：消除大盤隨機漫步的雜訊
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=TrainConfig.N_SPLITS)
    val_aucs = []

    for step, (train_index, val_index) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # 崩盤標籤極度稀少，若某個切片沒有崩盤日，給予 0.5 基準分跳過，防止報錯
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            val_aucs.append(0.5)
            continue

        # 動態計算正負樣本權重
        pos_count = max(sum(y_tr == 1), 1)
        neg_count = max(sum(y_tr == 0), 1)
        scale_pos_weight = neg_count / pos_count

        model = lgb.LGBMClassifier(**param, scale_pos_weight=scale_pos_weight)

        callbacks = [lgb.early_stopping(stopping_rounds=TrainConfig.EARLY_STOP_ROUND, verbose=False)]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        val_aucs.append(auc)

        trial.report(np.mean(val_aucs), step)
        if step >= 2 and trial.should_prune():
            raise optuna.TrialPruned()

    if not val_aucs: return 0.0

    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)

    # 大盤防護罩極度重視穩定度，稍微加重變異懲罰
    final_score = mean_auc - (0.6 * std_auc)

    trial.set_user_attr("mean_auc", float(mean_auc))
    trial.set_user_attr("std_auc", float(std_auc))

    return final_score

def run_market_optimization():
    print("="*70)
    print("🛡️ LightGBM 大盤崩盤防禦模型 尋優引擎啟動")
    print("="*70)

    oos_days = 240
    lookahead = 5 # 配合您的設定

    # ================= 準備大盤資料 =================
    print("⏳ 正在萃取大盤特徵資料...")
    # 隨便用一檔權值股當作主體去拉資料即可，因為我們會拿到 MacroTicker
    engine = QuantAIEngine(ticker="0050.TW", oos_days=oos_days)
    macro_tickers = [e.value for e in MacroTicker]
    df_raw = engine.db.get_aligned_market_data("0050.TW", macro_tickers)

    df_train_raw = df_raw.iloc[:-oos_days] if oos_days > 0 else df_raw

    market_engine = MarketFeatureEngine(lookahead=lookahead)
    df_clean = market_engine.process_pipeline(df_train_raw, is_training=True)

    if df_clean.empty:
        print("❌ 無法取得大盤資料，請檢查資料庫。")
        return

    features = MarketFeatureCol.get_features()
    X_train = df_clean[features]
    y_train = df_clean[MarketFeatureCol.TARGET_DANGER].astype(int)

    print(f"✅ 大盤資料準備完成！總訓練樣本: {len(X_train)} 筆")

    # ================= Optuna 設定 =================
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log_path = PathConfig.RESULT_REPORT / "market_lgbm_optuna_study.log"
    storage = JournalStorage(JournalFileStorage(str(log_path)))

    study = optuna.create_study(direction="maximize", study_name="Market_LGBM_v1", storage=storage, load_if_exists=True)

    TARGET_TOTAL_TRIALS = 500 # 樹模型 500 次足夠
    remaining = max(0, TARGET_TOTAL_TRIALS - len(study.trials))

    if remaining > 0:
        with tqdm(total=remaining, desc="🎯 大盤調參", unit="trial") as pbar:
            def callback(study, trial):
                pbar.set_postfix({"Best Score": f"{study.best_value:.4f}"})
                pbar.update(1)
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=remaining, callbacks=[callback], n_jobs=-1)

    print("\n" + "="*70)
    print("🏆 【大盤尋優完成】最強防禦參數誕生！")
    print("="*70)
    if study.best_trial:
        print(f"🥇 最佳風險調整 AUC: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

if __name__ == "__main__":
    run_market_optimization()