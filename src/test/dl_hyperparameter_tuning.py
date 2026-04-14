import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.const import MacroTicker
from debug import dbg
from ml.const import DLModelType, RNNType
from ml.data.dl_features import DLFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import DLHyperParams
from ml.trainers.dl_trainer import DLTrainer
from path import PathConfig

MAX_CONSECUTIVE_FAILURES = 3
consecutive_failures = 0

def tune_global_hyperparameters(test_tickers: list, oos_days: int = 240):
    global consecutive_failures

    dbg.log("\n" + "="*60)
    dbg.log("🎯 啟動 Optuna 全域超參數尋優 (Global Hyperparameter Tuning)")
    dbg.log(f"   - 考場標的: {len(test_tickers)} 檔")
    dbg.log(f"   - OOS 隔離: {oos_days} 天")
    dbg.log("="*60)

    def objective(trial: optuna.Trial):
        global consecutive_failures

        suggested_channels = trial.suggest_categorical("CNN_OUT_CHANNELS", [8, 16, 32])
        suggested_hidden = trial.suggest_categorical("LSTM_HIDDEN", [16, 32])

        suggested_lr = trial.suggest_float("LEARNING_RATE", 1e-4, 5e-3, log=True)
        suggested_dropout = trial.suggest_float("DROPOUT", 0.2, 0.5)

        # 因為我們鎖定 n_jobs=1，這樣做百分之百安全，且不需要去動到任何底層的神經網路代碼！
        DLHyperParams.CNN_OUT_CHANNELS = suggested_channels
        DLHyperParams.LSTM_HIDDEN = suggested_hidden
        DLHyperParams.LEARNING_RATE = suggested_lr
        DLHyperParams.DROPOUT = suggested_dropout

        ticker_aucs = []

        for step, ticker in enumerate(test_tickers):
            try:
                engine = QuantAIEngine(ticker=ticker, oos_days=oos_days,
                                       dl_model_type=DLModelType.HYBRID, rnn_type=RNNType.LSTM)

                macro_tickers = [e.value for e in MacroTicker]
                df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

                if df_raw.empty:
                    continue

                df_train = df_raw.iloc[:-oos_days] if oos_days > 0 else df_raw

                # 特徵工程不受超參數影響，正常執行
                dl_engine = DLFeatureEngine(engine.config.lookahead)
                X_train, y_train, valid_index = dl_engine.process_pipeline(df_train, is_training=True)

                if X_train is None or len(X_train) < 50:
                    dbg.war(f"Trial {trial.number}: {ticker} 資料不足，觸發剪枝。")
                    raise optuna.TrialPruned()

                # 拿掉 custom_hp，恢復原本單純的呼叫方式
                trainer = DLTrainer(ticker, DLModelType.HYBRID, RNNType.LSTM)
                oof_preds = trainer.train_with_cv(X_train, y_train, valid_index, engine.config.lookahead)

                if oof_preds.empty:
                    dbg.war(f"Trial {trial.number}: {ticker} 預測輸出為空，觸發剪枝。")
                    raise optuna.TrialPruned()

                y_true_series = pd.Series(y_train, index=valid_index)
                df_eval = pd.DataFrame({'prob': oof_preds, 'true': y_true_series}).dropna()

                if df_eval['true'].nunique() < 2:
                    dbg.war(f"Trial {trial.number}: {ticker} 標籤不平衡，觸發剪枝。")
                    raise optuna.TrialPruned()

                auc = roc_auc_score(df_eval['true'], df_eval['prob'])
                ticker_aucs.append(auc)

                trial.report(auc, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                consecutive_failures += 1
                dbg.error(f"❌ Trial {trial.number} 處理 {ticker} 時發生異常: {e}")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    dbg.error(f"🚨 連續發生 {MAX_CONSECUTIVE_FAILURES} 次致命錯誤！為保護系統，強制中止尋優！")
                    raise RuntimeError("連續錯誤次數過多，停止 Optuna。請檢查資料庫或記憶體狀態。") from e

                raise optuna.TrialPruned()

            finally:
                if 'engine' in locals(): del engine
                if 'trainer' in locals(): del trainer
                if 'X_train' in locals(): del X_train

                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        consecutive_failures = 0

        if not ticker_aucs:
            return 0.0

        mean_auc = np.mean(ticker_aucs)
        std_auc = np.std(ticker_aucs)

        penalty_factor = 0.5
        final_score = mean_auc - (penalty_factor * std_auc)

        trial.set_user_attr("mean_auc", float(mean_auc))
        trial.set_user_attr("std_auc", float(std_auc))

        return final_score
    # 在您的目錄下建立一個 sqlite 資料庫檔案
    sqlite_path = PathConfig.RESULT_REPORT / "optuna_tuning.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{sqlite_path.absolute()}"

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        study_name="IDSS_Hybrid_LSTM_Tuning",
        storage=storage_url,       # 使用 SQLite
        load_if_exists=True,       # 允許中斷後重新執行，接續之前的進度！
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    dbg.log(f"🚀 開始尋優！紀錄存於: {sqlite_path.name}")
    target_trials = 50
    current_trials = len(study.trials)
    remaining_trials = target_trials - current_trials

    if remaining_trials > 0:
        dbg.log(f"目前資料庫已有 {current_trials} 筆紀錄，準備補跑剩餘的 {remaining_trials} 筆...")
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True, n_jobs=1)
    else:
        dbg.log(f"資料庫中已經有 {current_trials} 筆紀錄，已達到或超過目標 {target_trials} 筆，不需再跑！")

    dbg.log("\n" + "="*60)
    dbg.log("🏆 Optuna 尋優完成！")
    dbg.log(f"最高評分 (Mean AUC - Penalty): {study.best_value:.4f}")
    dbg.log("請將以下最佳參數手動更新至您的 `ml/params.py`：")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            dbg.log(f"    {key} = {value:.6f}")
        else:
            dbg.log(f"    {key} = {value}")
    dbg.log("="*60)

if __name__ == "__main__":
    test_tickers = ["2330.TW", "2603.TW", "2881.TW", "2409.TW", "2344.TW", "2388.TW"]

    tune_global_hyperparameters(test_tickers=test_tickers)
