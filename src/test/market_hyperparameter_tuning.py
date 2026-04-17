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


# 定義一個簡單的代理類別
class LGBMLoggerProxy:
    def info(self, msg):
        pass
    def warning(self, msg):
        dbg.war(f"[LGBM] {msg}")
    def error(self, msg):
        dbg.error(f"[LGBM] {msg}")

# 註冊這個代理物件
lgb.register_logger(LGBMLoggerProxy())

dbg.toggle()

def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series):
    """大盤防禦專屬目標函數 (極簡主義 + 高度正則化)"""

    # 核心：嚴格限制樹的複雜度
    max_depth = trial.suggest_int('max_depth', 2, 5) # 大盤資料少，最多只能切 2~5 刀

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,      # 讓 Optuna 管多核
        'verbose': -1,
        'n_estimators': 1000,

        'max_depth': max_depth,
        'num_leaves': trial.suggest_int('num_leaves', 3, int(2**max_depth) - 1),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),

        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'max_bin': trial.suggest_categorical('max_bin', [31, 63, 127, 255]),

        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),

        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),

        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=TrainConfig.N_SPLITS)
    val_aucs = []

    for step, (train_index, val_index) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            fallback_score = np.mean(val_aucs) if val_aucs else 0.5
            val_aucs.append(fallback_score)
            continue

        pos_count = max(sum(y_tr == 1), 1)
        neg_count = max(sum(y_tr == 0), 1)
        scale_pos_weight = neg_count / pos_count

        # 確保參數字典乾淨無衝突
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

def run_market_optimization(oos_days: int, lookahead: int, target_total_trials: int):
    print("="*70)
    print("🛡️ LightGBM 大盤崩盤防禦模型 尋優引擎啟動")
    print("="*70)

    print("⏳ 正在萃取大盤特徵資料...")
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log_path = PathConfig.RESULT_REPORT / "market_lgbm_optuna_study.log"
    storage = JournalStorage(JournalFileStorage(str(log_path)))

    study = optuna.create_study(direction="maximize", study_name="Market_LGBM_v1", storage=storage, load_if_exists=True)

    completed_trials = len(study.trials)
    remaining_trials = max(0, target_total_trials - completed_trials)

    if remaining_trials == 0:
        print(f"✅ 尋優專案已完成 {target_total_trials} 次測試，無需再跑！")
    else:
        print(f"⏳ 目前已完成 {completed_trials} 次，剩餘 {remaining_trials} 次測試即將開始...")

        with tqdm(total=remaining_trials, desc="🎯 大盤調參", unit="trial") as pbar:
            def callback(study, trial):
                pbar.set_postfix({"Best Score": f"{study.best_value:.4f}"})
                pbar.update(1)
            # 開啟多核加速
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=remaining_trials, callbacks=[callback], n_jobs=1)

    print("\n" + "="*70)
    print("🏆 【大盤尋優完成】最強防禦參數誕生！")
    print("="*70)
    if len(study.trials) > 0 and study.best_trial:
        print(f"🥇 最佳風險調整 AUC: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

if __name__ == "__main__":
    from ml.params import SessionConfig
    lookahead = SessionConfig.lookahead
    oos_days = 240
    # 因為是大盤所以只跑 500 次
    target_total_trials = 500
    run_market_optimization(oos_days=oos_days, lookahead=lookahead, target_total_trials=target_total_trials)