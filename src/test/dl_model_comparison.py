import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from bt.backtest import BacktestEngine
from bt.strategy_config import PersonaFactory, TradingPersona
from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import DLModelType, FeatureCol, RNNType, SignalCol
from ml.data.xgb_features import XGBFeatureEngine
from ml.engine import QuantAIEngine
from path import PathConfig

dbg.toggle()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def run_model_comparison(test_tickers: list, init_cash: int, oos_days: int = 240):
    set_seed(42)

    model_configs = [
        {"name": "Pure_1D_CNN", "dl_type": DLModelType.PURE_CNN, "rnn": None},
        {"name": "Hybrid_LSTM", "dl_type": DLModelType.HYBRID, "rnn": RNNType.LSTM},
        {"name": "Hybrid_GRU",  "dl_type": DLModelType.HYBRID, "rnn": RNNType.GRU}
    ]

    strategy = PersonaFactory.get_config(TradingPersona.MODERATE)

    experiment_results = []

    # 分成兩個輸出檔案更清晰
    details_path = Path(PathConfig.EXPERIMENT_DETAILS)
    summary_path = Path(PathConfig.EXPERIMENT_SUMMARY)

    print("\n" + "="*60)
    print("🚀 IDSS 深度學習架構 A/B 測試啟動 (含 ML 核心指標)")
    print("="*60)

    for ticker in test_tickers:
        print(f"\n\n{'='*50}\n📊 開始處理標的: {ticker}\n{'='*50}")

        for config in model_configs:
            model_name = config["name"]
            dl_type = config["dl_type"]
            rnn_type = config["rnn"]

            print(f"\n🧪 正在測試架構: 【{model_name}】")

            # 提前宣告變數，防止 finally 清理時發生 UnboundLocalError
            ai_engine = None
            bt_engine = None
            df_raw = None
            df_with_target = None

            try:
                ai_engine = QuantAIEngine(
                    ticker=ticker,
                    oos_days=oos_days,
                    api_keys=None,
                    dl_model_type=dl_type,
                    rnn_type=rnn_type
                )

                ai_engine.update_market_data()
                start_time = time.time()
                ai_engine.train_all_models(save_models=True)

                if not ai_engine.load_inference_models():
                    dbg.error(f"❌ {model_name} 載入失敗，跳過...")
                    continue

                df_real_data = ai_engine.generate_backtest_data()
                if df_real_data.empty:
                    continue
                df_test = df_real_data.tail(oos_days).copy()

                macro_tickers = [e.value for e in MacroTicker]
                df_raw = ai_engine.db.get_aligned_market_data(ticker, macro_tickers)

                xgb_engine = XGBFeatureEngine()
                df_with_target = xgb_engine.process_pipeline(df_raw, lookahead=ai_engine.config.lookahead, is_training=True)

                df_test = df_test.join(df_with_target[[FeatureCol.TARGET.value]], how='left')
                df_eval = df_test.dropna(subset=[FeatureCol.TARGET.value, SignalCol.PROB_DL.value, SignalCol.PROB_FINAL.value])

                dl_auc, dl_acc, final_auc, final_acc = 0.0, 0.0, 0.0, 0.0

                if not df_eval.empty and df_eval[FeatureCol.TARGET.value].nunique() > 1:
                    y_true = df_eval[FeatureCol.TARGET.value]

                    y_dl_prob = df_eval[SignalCol.PROB_DL.value]
                    dl_auc = roc_auc_score(y_true, y_dl_prob)
                    dl_acc = accuracy_score(y_true, (y_dl_prob > 0.5).astype(int))

                    y_final_prob = df_eval[SignalCol.PROB_FINAL.value]
                    final_auc = roc_auc_score(y_true, y_final_prob)
                    final_acc = accuracy_score(y_true, (y_final_prob > 0.5).astype(int))

                bt_engine = BacktestEngine(initial_cash=init_cash, ticker=ticker, strategy=strategy)
                stats = bt_engine.run(df=df_test, silence=True)
                train_time = time.time() - start_time

                first_close = df_test[StockCol.CLOSE.value].iloc[0]
                last_close = df_test[StockCol.CLOSE.value].iloc[-1]
                bnh_return = (last_close - first_close) / first_close

                record = {
                    "Ticker": ticker,
                    "Model": model_name,
                    "DL_AUC": round(dl_auc, 4),
                    "DL_Acc(%)": round(dl_acc * 100, 2),
                    "Meta_AUC": round(final_auc, 4),
                    "Meta_Acc(%)": round(final_acc * 100, 2),
                    "B&H_Return(%)": round(bnh_return * 100, 2),
                    "Return(%)": round(stats.get("total_return", 0) * 100, 2) if stats else 0.0,
                    "MDD(%)": round(stats.get("mdd", 0) * 100, 2) if stats else 0.0,
                    "Sharpe": round(stats.get("sharpe", 0), 2) if stats else 0.0,
                    "Time(s)": round(train_time, 1)
                }
                experiment_results.append(record)

                print(f"✅ {model_name} | DL_AUC: {record['DL_AUC']} | 報酬率: {record['Return(%)']}% | MDD: {record['MDD(%)']}%")

                # 即時寫入詳細報告，避免中斷時資料遺失
                df_results = pd.DataFrame(experiment_results)

                # 確保父資料夾存在
                details_path.parent.mkdir(parents=True, exist_ok=True)
                df_results.to_csv(details_path, index=False, encoding="utf-8-sig")

            except Exception as e:
                dbg.error(f"❌ 處理 {ticker} - {model_name} 時發生嚴重錯誤: {e}")

            finally:
                # 安全的記憶體釋放
                del ai_engine
                del bt_engine
                del df_raw
                del df_with_target

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    if not experiment_results:
        dbg.error("沒有收集到任何結果！")
        return

    df_results = pd.DataFrame(experiment_results)

    numeric_cols = ['DL_AUC', 'Meta_AUC', 'B&H_Return(%)', 'Return(%)', 'MDD(%)', 'Sharpe', 'Time(s)']
    df_summary = df_results.groupby('Model')[numeric_cols].mean().round(3).reset_index()

    # 確保父資料夾存在
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n🎉 實驗完成！")
    print(f"📄 各標的詳細報表已輸出至: {details_path}")
    print(f"🏆 模型綜合戰力總結已輸出至: {summary_path}")

if __name__ == "__main__":
    test_tickers = [
        "3006.TW", "4916.TW", "9958.TW", "2481.TW",
        "2337.TW", "3563.TW", "2313.TW", "4919.TW"
    ]
    init_cash = 2_000_000

    run_model_comparison(test_tickers=test_tickers, init_cash=init_cash)
