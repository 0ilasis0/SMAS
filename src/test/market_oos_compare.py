import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)

from data.const import MacroTicker
from debug import dbg
from ml.const import MarketFeatureCol
from ml.data.market_features import MarketFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import TrainConfig
# ================= 引入您的實際生產模組 =================
from path import PathConfig


class LGBMLoggerProxy:
    def info(self, msg):
        pass
    def warning(self, msg):
        dbg.war(f"[LGBM] {msg}")
    def error(self, msg):
        dbg.error(f"[LGBM] {msg}")

lgb.register_logger(LGBMLoggerProxy())

def run_market_comparison(lookahead: int, oos_days: int = 240):
    print("="*70)
    print("🕵️‍♂️ LightGBM 大盤崩盤防禦模型 (OOS) 盲測對照引擎")
    print("="*70)

    # ================= 1. 定義對照組與實驗組參數 =================
    # 🔴 對照組 (Baseline)：預設參數，不使用 early_stopping，固定跑 100 棵樹
    BASELINE_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,

        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # 🟢 實驗組 (Optimized)：請填入您 Optuna 跑出的最佳結果
    OPTIMIZED_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,

        # --- ⬇️ 請替換為您的 Optuna 最新結果 ⬇️ ---
        'n_estimators': 100,
        'max_depth': 3,
        'num_leaves': 4,
        'min_child_samples': 17,
        'min_split_gain': 3.8681,
        'learning_rate':  0.0030,
        'subsample': 0.7371,
        'feature_fraction': 0.4169,
        'reg_alpha': 2.3999,
        'reg_lambda': 0.1388,
        'max_bin': 255
    }

    print("⏳ 正在萃取大盤特徵與準備盲測資料...")

    # ================= 2. 準備大盤資料與防洩漏切分 =================
    try:
        engine = QuantAIEngine(ticker="0050.TW", oos_days=oos_days)
        macro_tickers = [e.value for e in MacroTicker]
        df_raw = engine.db.get_aligned_market_data("0050.TW", macro_tickers)

        market_engine = MarketFeatureEngine(lookahead=lookahead)
        df_clean = market_engine.process_pipeline(df_raw, is_training=True)

        if df_clean.empty:
            print("❌ 無法取得大盤資料，請檢查資料庫。")
            return

        # 防堵資料洩漏的「時空隔離帶 (Purge Gap)」
        df_train_full = df_clean.iloc[: -(oos_days + lookahead)]
        df_oos = df_clean.iloc[-oos_days:]

        features = MarketFeatureCol.get_features()
        X_train_full, y_train_full = df_train_full[features], df_train_full[MarketFeatureCol.TARGET_DANGER].astype(int)
        X_oos, y_oos = df_oos[features], df_oos[MarketFeatureCol.TARGET_DANGER].astype(int)

        if len(np.unique(y_oos)) < 2:
            print(f"⚠️ 警告：在最近的 {oos_days} 天盲測期內，大盤從未發生過 '崩盤(Danger=1)' 條件。")
            print("這將導致無法計算 AUC 與 Recall (因為沒有真實崩盤可以捕捉)。")
            # 視情況看您是否要繼續執行，這裡我們讓它繼續，但指標會是 0

        # 動態正負樣本權重 (大盤極度不平衡，這步非常重要)
        pos_count = max(sum(y_train_full == 1), 1)
        neg_count = max(sum(y_train_full == 0), 1)
        scale_pos_weight = neg_count / pos_count

        # ================= 3. 訓練 Baseline 模型 =================
        print("🧠 正在訓練 Baseline (對照組) 模型...")
        model_base = lgb.LGBMClassifier(**BASELINE_PARAMS, scale_pos_weight=scale_pos_weight)
        model_base.fit(X_train_full, y_train_full)

        preds_proba_base = model_base.predict_proba(X_oos)[:, 1]
        preds_label_base = (preds_proba_base >= 0.5).astype(int)

        # ================= 4. 訓練 Optimized 模型 (包含內部 CV 早停) =================
        print("🧠 正在訓練 Optuna Optimized (實驗組) 模型...")

        # 從訓練集內部切出最後 20% 當作 Valid Set，嚴格不看盲測集
        val_split_idx = int(len(X_train_full) * 0.8)
        X_tr, y_tr = X_train_full.iloc[:val_split_idx], y_train_full.iloc[:val_split_idx]
        X_val, y_val = X_train_full.iloc[val_split_idx:], y_train_full.iloc[val_split_idx:]

        model_opt = lgb.LGBMClassifier(**OPTIMIZED_PARAMS, scale_pos_weight=scale_pos_weight)

        # 確保 Validation Set 至少有包含崩盤標籤才能做早停
        if len(np.unique(y_val)) > 1:
            callbacks = [lgb.early_stopping(stopping_rounds=TrainConfig.EARLY_STOP_ROUND, verbose=False)]
            model_opt.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        else:
            print("⚠️ Validation 區間無崩盤事件，取消早停，退回固定樹量 200 棵...")
            opt_params_fallback = OPTIMIZED_PARAMS.copy()
            opt_params_fallback['n_estimators'] = 200
            model_opt = lgb.LGBMClassifier(**opt_params_fallback, scale_pos_weight=scale_pos_weight)
            model_opt.fit(X_train_full, y_train_full)

        preds_proba_opt = model_opt.predict_proba(X_oos)[:, 1]
        preds_label_opt = (preds_proba_opt >= 0.5).astype(int)

        # ================= 5. 評估並產生報告 =================
        def get_metrics(y_true, y_pred, y_prob):
            if len(np.unique(y_true)) < 2: return 0.5, 0, 0, 0
            return (
                roc_auc_score(y_true, y_prob),
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, zero_division=0),
                recall_score(y_true, y_pred, zero_division=0) # 崩盤捕獲率
            )

        auc_b, acc_b, prec_b, rec_b = get_metrics(y_oos, preds_label_base, preds_proba_base)
        auc_o, acc_o, prec_o, rec_o = get_metrics(y_oos, preds_label_opt, preds_proba_opt)

        report_data = [{
            "Index": "TWII_Market_Danger",
            "AUC_Before": round(auc_b, 4),
            "AUC_After": round(auc_o, 4),
            "AUC_Diff": round(auc_o - auc_b, 4),
            "Recall_Before(%)": round(rec_b * 100, 2),
            "Recall_After(%)": round(rec_o * 100, 2),
            "Prec_Before(%)": round(prec_b * 100, 2),
            "Prec_After(%)": round(prec_o * 100, 2),
            "Opt_Trees": model_opt.best_iteration_ if hasattr(model_opt, 'best_iteration_') and model_opt.best_iteration_ else "N/A"
        }]

        df_report = pd.DataFrame(report_data)
        PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
        csv_path = PathConfig.RESULT_REPORT / "market_lgbm_comparison_report.csv"
        df_report.to_csv(csv_path, index=False)

        # ================= 6. 終極報表印出 =================
        print("\n" + "="*70)
        print("🏆 LightGBM 大盤盲測績效總結")
        print("="*70)

        print(f"📉 調整前 (Baseline) 排序能力 (AUC): \t{auc_b:.4f}")
        print(f"🚀 調整後 (Optuna)   排序能力 (AUC): \t{auc_o:.4f} (差異: {auc_o - auc_b:+.4f})")
        print("-" * 70)

        print(f"🛡️ 崩盤捕獲率 (Recall) - 發生大跌時，雷達有響的機率：")
        print(f"   調整前 Baseline: {rec_b:.2%}")
        print(f"   調整後 Optuna:   {rec_o:.2%} (差異: {rec_o - rec_b:+.2%})")
        print("-" * 70)

        print(f"🎯 報警精準度 (Precision) - 雷達響起時，真的跌了的機率：")
        print(f"   調整前 Baseline: {prec_b:.2%}")
        print(f"   調整後 Optuna:   {prec_o:.2%}")
        print("="*70)

        print(f"🚨 OOS 期間 (過去 {oos_days} 天) 實際發生崩盤的天數: {sum(y_oos == 1)} 天")
        print(f"📁 詳細對照 CSV 已儲存至: {csv_path}")

    except Exception as e:
        print(f"\n⚠️ 評估失敗: {e}")

if __name__ == "__main__":
    lookahead = 20

    run_market_comparison(lookahead=lookahead)