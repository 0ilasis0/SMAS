import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from bt.backtest import BacktestEngine
from bt.strategy_config import PersonaFactory, TradingPersona
from debug import dbg
from ml.const import DLModelType, FeatureCol, MetaCol, RNNType
from ml.engine import QuantAIEngine
from path import PathConfig


def run_model_comparison():
    # ==========================================
    # 1. 實驗參數設定
    # ==========================================
    tickers = ["2330.TW", "2603.TW", "2881.TW", "2409.TW", "2344.TW", "2388.TW"]

    # 定義要 PK 的三種兵工廠架構
    model_configs = [
        {"name": "Pure_1D_CNN", "dl_type": DLModelType.PURE_CNN, "rnn": None},
        {"name": "Hybrid_LSTM", "dl_type": DLModelType.HYBRID, "rnn": RNNType.LSTM},
        {"name": "Hybrid_GRU",  "dl_type": DLModelType.HYBRID, "rnn": RNNType.GRU}
    ]

    test_days = 240  # 回測過去一整年 (240個交易日) 的純淨未參與訓練資料
    user_cash = 2000000
    strategy = PersonaFactory.get_config(TradingPersona.MODERATE) # 統一用穩健型人格測試

    experiment_results = []
    output_path = Path(PathConfig.EXPERIMENT_RESULTS)

    dbg.log("\n" + "="*60)
    dbg.log("🚀 IDSS 深度學習架構 A/B 測試啟動")
    dbg.log("="*60)

    # ==========================================
    # 2. 自動化實驗迴圈
    # ==========================================
    for ticker in tickers:
        dbg.log(f"\n\n{'='*50}\n📊 開始處理標的: {ticker}\n{'='*50}")

        for config in model_configs:
            model_name = config["name"]
            dl_type = config["dl_type"]
            rnn_type = config["rnn"]

            dbg.log(f"\n🧪 正在測試架構: 【{model_name}】")

            start_time = time.time()

            # 實例化 AI 引擎，動態注入模型架構
            ai_engine = QuantAIEngine(
                ticker=ticker,
                oos_days=test_days,
                api_keys=None, # 回測不需要消耗 Gemini API 寫戰報
                dl_model_type=dl_type,
                rnn_type=rnn_type
            )

            # 更新資料與強制重新訓練模型 (確保不同架構能獨立學習)
            ai_engine.update_market_data()
            ai_engine.train_all_models(save_models=True)

            if not ai_engine.load_inference_models():
                dbg.error(f"❌ {model_name} 載入失敗，跳過此架構...")
                continue

            # 產生包含機率的回測資料
            df_real_data = ai_engine.generate_backtest_data()
            if df_real_data.empty:
                dbg.war(f"⚠️ {ticker} - {model_name} 產生的回測資料為空，跳過...")
                continue

            # 切割出最後 test_days 天作為 OOS 測試區間
            df_test = df_real_data.tail(test_days).copy()

            # 嘗試計算 OOS 區間的 AUC (如果資料表中有包含 TARGET 欄位)
            oos_auc = "N/A"
            if FeatureCol.TARGET in df_test.columns and df_test[FeatureCol.TARGET].nunique() > 1:
                try:
                    oos_auc = roc_auc_score(df_test[FeatureCol.TARGET], df_test[MetaCol.PROB_FINAL])
                    oos_auc = round(oos_auc, 4)
                except Exception:
                    pass

            # 執行資金部位與行為樹回測
            bt_engine = BacktestEngine(initial_cash=user_cash, ticker=ticker, strategy=strategy)
            stats = bt_engine.run(df_test)

            train_time = time.time() - start_time

            # ==========================================
            # 3. 收集數據
            # ==========================================
            record = {
                "Ticker": ticker,
                "Model_Arch": model_name,
                "Test_Days": len(df_test),
                "OOS_AUC": oos_auc, # 樣本外預測準確度指標
                "Total_Return(%)": round(stats.get("total_return", 0) * 100, 2),
                "CAGR(%)": round(stats.get("cagr", 0) * 100, 2),
                "MDD(%)": round(stats.get("mdd", 0) * 100, 2), # 最大回撤 (風險指標)
                "Sharpe": round(stats.get("sharpe", 0), 2),    # 夏普值 (CP值指標)
                "Buy_Count": stats.get("buy_count", 0),
                "Sell_Count": stats.get("sell_count", 0),
                "Execution_Time(s)": round(train_time, 1) # 觀察 CNN 是不是比 LSTM 快很多
            }
            experiment_results.append(record)

            dbg.log(f"✅ {model_name} 測試完成 | 報酬率: {record['Total_Return(%)']}% | MDD: {record['MDD(%)']}% | 耗時: {record['Execution_Time(s)']}秒")

            # 即時存檔，避免跑到一半當機資料遺失
            df_results = pd.DataFrame(experiment_results)
            df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ==========================================
    # 4. 輸出最終實驗結果
    # ==========================================
    dbg.log(f"\n🎉 所有實驗完成！比較報表已輸出至: {output_path}")
    print("\n" + "="*80)
    print("📈 模型架構深度比較結果 (Experiment Results)")
    print("="*80)
    print(df_results.to_markdown(index=False))


if __name__ == "__main__":
    run_model_comparison()
