import pandas as pd

from bt.backtest import BacktestEngine
from bt.strategy_config import PersonaFactory, TradingPersona
from debug import dbg
from ml.engine import QuantAIEngine

dbg.toggle()

def fetch_backtest_data(ticker: str, oos_days: int = 240) -> pd.DataFrame:
    """模擬 IDSSController 獲取回測資料的流程"""
    try:
        engine = QuantAIEngine(ticker=ticker, oos_days=oos_days)
        # 嘗試載入模型，如果失敗則直接回傳空 DataFrame
        if not engine.load_inference_models():
            print(f"⚠️ 找不到 {ticker} 的模型，跳過回測。")
            return pd.DataFrame()

        df = engine.generate_backtest_data()
        # 盲測期間關閉 LLM (避免觸發新聞 API)
        return df.tail(oos_days)
    except Exception as e:
        print(f"⚠️ {ticker} 獲取資料失敗: {e}")
        return pd.DataFrame()

def run_multi_stock_backtest():
    print("="*60)
    print("🚀 IDSS 混合制批量回測系統啟動")
    print("="*60)

    # 1. 定義測試池 (建議涵蓋不同股性)
    test_tickers = [
        "2330.TW",  # 台積電 (權值牛皮)
        "2409.TW",  # 友達 (景氣循環)
        "2388.TW",  # 威盛 (高波動妖股)
        "2324.TW",  # 仁寶 (低波高股息)
        "0050.TW"   # 台灣50 (大盤 ETF)
    ]

    # 設定每次回測的初始資金與天數
    INITIAL_CASH = 300_000
    OOS_DAYS = 240

    all_results = []

    # 2. 執行雙層迴圈 (股票 x 個性)
    for ticker in test_tickers:
        print(f"\n📥 正在準備 {ticker} 的回測資料...")
        df_test = fetch_backtest_data(ticker, oos_days=OOS_DAYS)
        if df_test.empty: continue

        print(f"📊 開始對 {ticker} 進行三種性格交叉測試：")
        for persona in [TradingPersona.AGGRESSIVE, TradingPersona.MODERATE, TradingPersona.CONSERVATIVE]:

            # 取得對應性格的參數
            strategy_config = PersonaFactory.get_config(persona)
            strategy_config.enable_llm_oracle = False # 盲測關閉 LLM

            # 初始化回測引擎
            engine = BacktestEngine(initial_cash=INITIAL_CASH, ticker=ticker, strategy=strategy_config)

            # 為了單純看數據，我們在 backtest engine 裡面不要印出每一天的 log
            stats = engine.run(df_test, silence=True)

            if stats:
                result_row = {
                    "Ticker": ticker,
                    "Persona": persona.value,
                    "Return (%)": round(stats['total_return'] * 100, 2),
                    "MDD (%)": round(stats['mdd'] * 100, 2),
                    "Sharpe": round(stats['sharpe'], 2),
                    "Trades": stats['buy_count'] + stats['sell_count']
                }
                all_results.append(result_row)
                print(f"   [{persona.value.upper()}] 報酬: {result_row['Return (%)']:>6}%, MDD: {result_row['MDD (%)']:>6}%, Sharpe: {result_row['Sharpe']:>4}")

    # 3. 彙整與分析報告
    if not all_results:
        print("\n❌ 所有測試皆失敗，請確認模型是否已訓練。")
        return

    df_report = pd.DataFrame(all_results)

    print("\n\n" + "="*60)
    print("🏆 多標的批量回測完整報告")
    print("="*60)
    # 將 Pandas 輸出格式化，對齊好看
    print(df_report.to_string(index=False))
    print("\n")

    # 4. 計算每種性格在「所有股票」上的平均表現 (尋找真正 Robust 的參數)
    print("🎯 【綜合性格評比 (平均表現)】")
    summary = df_report.groupby("Persona").agg({
        "Return (%)": "mean",
        "MDD (%)": "mean",
        "Sharpe": "mean",
        "Trades": "mean"
    }).reset_index()

    print(summary.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    run_multi_stock_backtest()