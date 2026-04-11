import time

import pandas as pd

from bt.backtest import BacktestEngine
from bt.strategy_config import PersonaFactory, TradingPersona
from debug import dbg
from ml.engine import QuantAIEngine
from path import PathConfig

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

    PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)

    # 1. 定義測試池
    test_tickers = [
        "2330.TW", "2409.TW", "2388.TW", "2324.TW", "0050.TW",
        "3481.TW", "0052.TW", "2481.TW", "2603.TW", "2881.TW",
        "2344.TW", "4919.TW", "3231.TW", "2455.TW", "9958.TW",
        "3006.TW", "2301.TW", "4916.TW", "2317.TW"
    ]

    INITIAL_CASH = 5_000_000
    OOS_DAYS = 240
    all_results = []

    # 2. 執行雙層迴圈 (股票 x 個性)
    for ticker in test_tickers:
        print(f"\n📥 正在準備 {ticker} 的回測資料...")
        df_test = fetch_backtest_data(ticker, oos_days=OOS_DAYS)
        if df_test.empty: continue

        print(f"📊 開始對 {ticker} 進行三種性格交叉測試：")

        for persona in [TradingPersona.AGGRESSIVE, TradingPersona.MODERATE, TradingPersona.CONSERVATIVE]:
            strategy_config = PersonaFactory.get_config(persona)
            strategy_config.enable_llm_oracle = False

            engine = BacktestEngine(initial_cash=INITIAL_CASH, ticker=ticker, strategy=strategy_config)
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

    # 3. 彙整與分析總報告
    if not all_results:
        print("\n❌ 所有測試皆失敗，無法產生報告。")
        return

    df_report = pd.DataFrame(all_results)

    # 利用 Pandas 的排序功能，讓同一個股的性格緊密靠在一起
    persona_order = [TradingPersona.AGGRESSIVE.value, TradingPersona.MODERATE.value, TradingPersona.CONSERVATIVE.value]
    df_report['Persona'] = pd.Categorical(df_report['Persona'], categories=persona_order, ordered=True)

    # 先依據股票代碼排序，再依據我們定義的性格順序排序
    df_report = df_report.sort_values(by=['Ticker', 'Persona'])

    print("\n\n" + "="*60)
    print("🏆 多標的批量回測完整報告")
    print("="*60)
    print(df_report.to_string(index=False))

    all_details_path = PathConfig.ALL_STOCKS_PERSONA
    try:
        df_report.to_csv(all_details_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 所有股票詳細回測明細已儲存至: {all_details_path}")
    except PermissionError:
        fallback_name = f"all_stocks_persona_report_{int(time.time())}.csv"
        fallback_path = all_details_path.parent / fallback_name
        df_report.to_csv(fallback_path, index=False, encoding='utf-8-sig')
        print(f"\n⚠️ 警告：原檔案被鎖定 (可能正用 Excel 開啟)。")
        print(f"💾 已自動轉存至備用檔案: {fallback_path}")

    # 4. 計算平均表現
    print("\n🎯 【綜合性格評比 (平均表現)】")
    summary = df_report.groupby("Persona", observed=False).agg({
        "Return (%)": "mean",
        "MDD (%)": "mean",
        "Sharpe": "mean",
        "Trades": "mean"
    }).reset_index()

    print(summary.to_string(index=False))

    summary_path = PathConfig.SUMMARY_PERSONA
    try:
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"💾 綜合評比平均報告已儲存至: {summary_path}")
    except PermissionError:
        fallback_name = f"summary_persona_{int(time.time())}.csv"
        fallback_path = summary_path.parent / fallback_name
        summary.to_csv(fallback_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 警告：原檔案被鎖定。已自動轉存至: {fallback_path}")

    print("="*60)

if __name__ == "__main__":
    run_multi_stock_backtest()