import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

# 引入您的系統元件
from data.const import MacroTicker
from debug import dbg
from ml.const import DLModelType, RNNType
from ml.data.dl_features import DLFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import DLHyperParams
from ml.trainers.dl_trainer import DLTrainer

# 為了避免控制台被幾百個模型的 log 洗版，我們可以暫時降低 debug 的輸出層級
# (視您的 debug.py 實作而定，這裡假設保留重要訊息)

def tune_global_hyperparameters():
    # ==========================================
    # 1. 嚴格的環境設定
    # ==========================================
    # 挑選 6 檔涵蓋權值、金融、循環、中小型動能的標的作為「全域考場」
    test_tickers = ["2330.TW", "2603.TW", "2881.TW", "2409.TW", "2344.TW", "2388.TW"]

    # 絕對盲測區間 (Optuna 尋優時，絕對不能看到這 240 天的資料)
    oos_days = 240

    dbg.log("\n" + "="*60)
    dbg.log("🎯 啟動 Optuna 全域超參數尋優 (Global Hyperparameter Tuning)")
    dbg.log(f"   - 考場標的: {len(test_tickers)} 檔")
    dbg.log(f"   - OOS 隔離: {oos_days} 天")
    dbg.log("="*60)

    # ==========================================
    # 2. 定義 Optuna 的目標函數 (Objective)
    # ==========================================
    def objective(trial: optuna.Trial):
        # 讓 Optuna 嘗試不同的參數組合
        suggested_channels = trial.suggest_categorical("CNN_OUT_CHANNELS", [16, 32, 64])
        suggested_hidden = trial.suggest_categorical("LSTM_HIDDEN", [16, 32, 64, 128])
        suggested_lr = trial.suggest_float("LEARNING_RATE", 1e-4, 5e-3, log=True)
        suggested_dropout = trial.suggest_float("DROPOUT", 0.1, 0.5)

        # 🚀 覆寫全域超參數 (Hack: 直接修改 DLHyperParams 類別的屬性)
        DLHyperParams.CNN_OUT_CHANNELS = suggested_channels
        DLHyperParams.LSTM_HIDDEN = suggested_hidden
        DLHyperParams.LEARNING_RATE = suggested_lr
        DLHyperParams.DROPOUT = suggested_dropout

        ticker_aucs = []

        # 在同一個 Trial 中，連續挑戰所有股票
        for step, ticker in enumerate(test_tickers):
            try:
                # 實例化引擎 (設定 oos_days 以隔離資料)
                engine = QuantAIEngine(ticker=ticker, oos_days=oos_days,
                                       dl_model_type=DLModelType.HYBRID, rnn_type=RNNType.LSTM)

                # 取得已對齊大盤的歷史資料
                macro_tickers = [e.value for e in MacroTicker]
                df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

                if df_raw.empty:
                    continue

                # 切割出訓練集 (嚴格剔除 OOS 天數)
                df_train = df_raw.iloc[:-oos_days] if oos_days > 0 else df_raw

                # 進行 DL 特徵工程
                dl_engine = DLFeatureEngine(engine.config.lookahead)
                X_train, y_train, valid_index = dl_engine.process_pipeline(df_train, is_training=True)

                if X_train is None or len(X_train) < 50:
                    raise optuna.TrialPruned() # 資料不足，直接剪枝砍掉

                # 呼叫訓練器執行 5-Fold CV
                trainer = DLTrainer(ticker, DLModelType.HYBRID, RNNType.LSTM)
                oof_preds = trainer.train_with_cv(X_train, y_train, valid_index, engine.config.lookahead)

                if oof_preds.empty:
                    raise optuna.TrialPruned()

                # 計算該股票的 OOF AUC
                y_true_series = pd.Series(y_train, index=valid_index)
                df_eval = pd.DataFrame({'prob': oof_preds, 'true': y_true_series}).dropna()

                if df_eval['true'].nunique() < 2:
                    raise optuna.TrialPruned() # 標籤不平衡，無法算 AUC

                auc = roc_auc_score(df_eval['true'], df_eval['prob'])
                ticker_aucs.append(auc)

                # 🚀 修剪機制 (Pruning)：
                # 如果這組參數在第一檔或第二檔股票就表現極差 (例如 AUC < 0.51)，直接中止這個 Trial，節省算力！
                trial.report(auc, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise # 傳遞修剪訊號
            except Exception as e:
                dbg.war(f"Trial {trial.number} 處理 {ticker} 時發生錯誤: {e}")
                raise optuna.TrialPruned()

        # 防呆
        if not ticker_aucs:
            return 0.0

        # ==========================================
        # 3. 核心：計算夏普式風險懲罰分數
        # ==========================================
        mean_auc = np.mean(ticker_aucs)
        std_auc = np.std(ticker_aucs)

        # 目標：平均 AUC 越高越好，但股票之間的表現差異 (std) 越小越好
        penalty_factor = 0.5
        final_score = mean_auc - (penalty_factor * std_auc)

        return final_score

    # ==========================================
    # 4. 啟動 Optuna 實驗室
    # ==========================================
    # 使用 MedianPruner：如果某個 Step (某檔股票) 的表現差於過去所有 Trial 的中位數，就直接剪枝
    study = optuna.create_study(
        direction="maximize",
        study_name="IDSS_Hybrid_LSTM_Tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    dbg.log("🚀 開始尋優！建議放著讓它跑幾個小時...")

    # 執行 50 個回合 (Trials)。您可以依據硬體速度調整 n_trials
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # ==========================================
    # 5. 輸出最終戰報
    # ==========================================
    dbg.log("\n" + "="*60)
    dbg.log("🏆 Optuna 尋優完成！")
    dbg.log(f"最高評分 (Mean AUC - Penalty): {study.best_value:.4f}")
    dbg.log("請將以下最佳參數手動更新至您的 `ml/params.py` 的 `DLHyperParams` 類別中：")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            dbg.log(f"    {key} = {value:.6f}")
        else:
            dbg.log(f"    {key} = {value}")
    dbg.log("="*60)

if __name__ == "__main__":
    tune_global_hyperparameters()
