from tqdm import tqdm

from data.const import MacroTicker
from debug import dbg
from ml.data.xgb_features import XGBFeatureEngine
from ml.engine import QuantAIEngine
from ml.params import XGBHyperParams
from ml.trainers.xgb_trainer import XGBTrainer
from path import PathConfig

dbg.toggle()

def update_xgb_backtest_models(train_tickers: list, lookahead: int, oos_days: int = 240):
    print("="*70)
    print("🚀 [回測環境] XGBoost 個股獨立模型更新程序啟動")
    print("="*70)

    # ================= 直接導入您的 XGBHyperParams (單一真理來源) =================
    hp = XGBHyperParams()

    print("📝 載入系統最新超參數設定：")
    for k, v in hp.__dict__.items():
        print(f"  {k}: {v}")

    print(f"\n⏳ 正在為 {len(train_tickers)} 檔標的『獨立訓練並更新模型』...")

    # ================= 逐檔股票獨立訓練與存檔 =================
    for ticker in tqdm(train_tickers, desc="模型更新進度"):
        try:
            # 獲取單檔股票資料
            engine = QuantAIEngine(ticker=ticker, oos_days=oos_days)
            macro_tickers = [e.value for e in MacroTicker]
            df_raw = engine.db.get_aligned_market_data(ticker, macro_tickers)

            if df_raw.empty:
                print(f"\n⚠️ {ticker} 查無資料，跳過。")
                continue

            # 特徵工程
            xgb_engine = XGBFeatureEngine()
            df_clean = xgb_engine.process_pipeline(df_raw, lookahead=lookahead, is_training=True)

            # 防範資料洩漏：只取前段資料進行訓練，完全不看 oos_days 內的資料
            df_train = df_clean.iloc[: -(oos_days + lookahead)]

            # 實例化 XGBTrainer (使用該個股的 Ticker)
            trainer = XGBTrainer(ticker=ticker, hp=hp)
            trainer.optimal_trees = hp.n_estimators

            # 取得專屬該個股的存檔路徑
            save_path = PathConfig.get_xgboost_model_path(ticker=ticker, oos_days=oos_days)

            # 訓練並存檔
            trainer.train_and_save_final_model(df_clean=df_train, save_path=save_path)

        except Exception as e:
            print(f"\n⚠️ {ticker} 模型更新失敗: {e}")

    print("\n" + "="*70)
    print("🎉 更新成功！所有個股的回測模型均已更新完畢。")
    print(f"📁 儲存目錄: {PathConfig.MODEL_DIR}")
    print("="*70)

if __name__ == "__main__":
    train_tickers = [
        "2344.TW", "2455.TW", "3006.TW", "2301.TW",
        "2481.TW", "0052.TW", "4919.TW", "3481.TW",
        "3231.TW", "4916.TW", "2324.TW", "9958.TW",
        "2330.TW", "0050.TW", "2603.TW", "2317.TW", "2881.TW", "2409.TW", "2388.TW"
    ]

    lookahead = 20

    update_xgb_backtest_models(train_tickers=train_tickers, lookahead=lookahead)
