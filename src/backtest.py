import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bt.account import Account
from bt.blackboard import Blackboard
from bt.const import BtVar, DecisionAction
from bt.params import StrategyConfig
from bt.strategy import build_trading_tree
from data.const import StockCol
from debug import dbg
from ml.const import MetaCol


class BacktestEngine:
    """
    IDSS 行為樹專用回測引擎。
    結合歷史 K 線與 AI 預測勝率，逐日進行沙盤推演，並計算最終績效。
    """
    def __init__(self, initial_cash: int):
        self.initial_cash = initial_cash
        self.account = Account(cash=initial_cash)
        self.bb = Blackboard(ticker="BACKTEST", account=self.account)
        self.tree = build_trading_tree(StrategyConfig())

        self.history_records = []

    def run(self, df: pd.DataFrame):
        """
        執行回測。
        :param df: 必須包含 ['Close', 'High', MetaCol.PROB_XGB, MetaCol.PROB_DL, MetaCol.PROB_FINAL] 欄位，且 Index 為日期。
        """
        self.history_records.clear()

        # 將帳戶與黑板狀態重置為初始狀態
        self.account.cash = self.initial_cash
        self.bb.clear_trade_memory()

        dbg.log(f"🚀 開始執行行為樹回測，初始資金: {self.initial_cash:,.0f} 元，共 {len(df)} 個交易日...")

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            date = df.index[i]
            current_close = row[StockCol.CLOSE]

            # 將今天的收盤資訊與明天的「開盤價」、「成交量」傳給黑板
            self.bb.current_date = str(date)

            self.bb.update_price(
                current_price=current_close,
                high_price=row[StockCol.HIGH],
                executable_price=next_row[StockCol.OPEN],  # 實際執行交易的價格
                daily_volume=next_row[StockCol.VOLUME]     # 流動性上限
            )
            self.bb.prob_xgb = row[MetaCol.PROB_XGB]
            self.bb.prob_dl = row[MetaCol.PROB_DL]
            self.bb.prob_final = row[MetaCol.PROB_FINAL]

            # 清空前一天的決策紀錄
            self.bb.action_decision = DecisionAction.HOLD

            # 全域時鐘，每天確實扣減冷卻期
            current_cd = getattr(self.bb, BtVar.COOLDOWN_TIMER, 0)
            if current_cd > 0:
                setattr(self.bb, BtVar.COOLDOWN_TIMER, current_cd - 1)

            # 執行行為樹心跳 (Tick)
            self.tree.tick(self.bb)

            # 計算當日總淨值
            stock_value = self.bb.position * current_close
            total_equity = self.bb.cash + stock_value

            # 紀錄歷史
            self.history_records.append({
                'Date': date,
                'Close': current_close,
                'Cash': self.bb.cash,
                'Position': self.bb.position,
                'Total_Equity': total_equity,
                'Action': self.bb.action_decision,
                MetaCol.PROB_FINAL: self.bb.prob_final
            })

        self._generate_report()

    def _generate_report(self):
        """計算績效指標並繪製資金曲線"""
        if not self.history_records:
            dbg.error("沒有回測紀錄可供產出報告！")
            return

        df_res = pd.DataFrame(self.history_records).set_index('Date')

        # 總報酬率與年化報酬率 (CAGR)
        final_equity = df_res['Total_Equity'].iloc[-1]
        total_return = (final_equity - self.initial_cash) / self.initial_cash

        # 計算 CAGR (假設一年約 252 個交易日)
        trading_days = len(df_res)
        cagr = (final_equity / self.initial_cash) ** (252 / trading_days) - 1

        # 計算最大回撤 (Max Drawdown, MDD)
        df_res['Peak'] = df_res['Total_Equity'].cummax()
        df_res['Drawdown'] = (df_res['Total_Equity'] - df_res['Peak']) / df_res['Peak']
        max_drawdown = df_res['Drawdown'].min()

        # 計算夏普值 (Sharpe Ratio，假設無風險利率為 1%)
        df_res['Daily_Return'] = df_res['Total_Equity'].pct_change().fillna(0)
        daily_volatility = df_res['Daily_Return'].std()
        if daily_volatility > 0:
            sharpe_ratio = (df_res['Daily_Return'].mean() - (0.01 / 252)) / daily_volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 統計交易次數
        buy_count = len(df_res[df_res['Action'] == DecisionAction.BUY])
        sell_count = len(df_res[df_res['Action'] == DecisionAction.SELL])

        dbg.log("\n" + "="*40)
        dbg.log("📊 IDSS 行為樹回測績效報告")
        dbg.log("="*40)
        dbg.log(f"💰 初始資金: \t{self.initial_cash:,.0f} 元")
        dbg.log(f"🏦 最終淨值: \t{final_equity:,.0f} 元")
        dbg.log(f"📈 總報酬率: \t{total_return:.2%}")
        dbg.log(f"🚀 年化報酬(CAGR): \t{cagr:.2%}")
        dbg.log(f"📉 最大回撤(MDD): \t{max_drawdown:.2%}")
        dbg.log(f"⚖️ 夏普值(Sharpe): \t{sharpe_ratio:.2f}")
        dbg.log(f"🛒 買進次數: \t{buy_count} 次")
        dbg.log(f"💸 賣出次數: \t{sell_count} 次")
        dbg.log("="*40)

        # 繪製資金曲線圖
        plt.figure(figsize=(12, 6))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df_res.index, df_res['Total_Equity'], label='Total Equity', color='blue')
        ax1.set_title('IDSS Strategy Equity Curve')
        ax1.set_ylabel('NTD')
        ax1.grid(True)
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(df_res.index, df_res['Close'], label='Stock Price', color='gray', alpha=0.6)

        buys = df_res[df_res['Action'] == DecisionAction.BUY]
        sells = df_res[df_res['Action'] == DecisionAction.SELL]
        ax2.scatter(buys.index, buys['Close'], marker='^', color='green', s=80, label='Buy')
        ax2.scatter(sells.index, sells['Close'], marker='v', color='red', s=80, label='Sell')

        ax2.set_ylabel('Stock Price')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

# ==========================================
# 模擬測試區 (如果還沒有真實的 ML 預測數據，可以用這個測試行為樹邏輯)
# ==========================================
def generate_mock_data(days: int = 250) -> pd.DataFrame:
    """生成具有趨勢的假 K 線與假 AI 勝率，用來測試行為樹是否正常運作"""
    np.random.seed(42)

    # 模擬股價走勢 (Random Walk with Drift)
    returns = np.random.normal(0.001, 0.02, days)
    price = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Close': price,
        'High': price * np.random.uniform(1.0, 1.03, days)
    }, index=pd.date_range(start='2025-01-01', periods=days, freq='B'))

    # 模擬 AI 勝率：假設 AI 能稍微預測到未來的股價變化
    # 我們給它一點雜訊，讓它的勝率在 0.2 ~ 0.9 之間震盪
    df[MetaCol.PROB_XGB] = np.clip(0.5 + np.random.normal(0, 0.1, days) + (returns * 5), 0.1, 0.9)
    df[MetaCol.PROB_DL] = np.clip(0.5 + np.random.normal(0, 0.1, days) + (returns * 5), 0.1, 0.9)
    df[MetaCol.PROB_FINAL] = (df[MetaCol.PROB_XGB] + df[MetaCol.PROB_DL]) / 2

    # 手動創造一個極端暴跌，測試停損機制
    df.loc[df.index[100:105], 'Close'] *= 0.8
    df.loc[df.index[100:105], MetaCol.PROB_FINAL] = 0.2 # 讓 AI 亮紅燈

    return df

if __name__ == "__main__":
    from ml.engine import QuantAIEngine

    ticker = "0052.TW"
    test_days = 240
    ai_engine = QuantAIEngine(ticker=ticker)

    # 假設你需要重新訓練模型 (如果模型已經是乾淨的，這段可以註解)
    # ai_engine.update_market_data()
    # ai_engine.train_all_models(save_models=True, oos_days=test_days)

    if not ai_engine.load_inference_models():
        dbg.error("❌ 模型載入失敗...")
        exit()

    # 此時 generate_backtest_data 會用「只學過過去」的模型，
    # 去預測「包含最後 240 天」的全量資料，這就是真正的 OOS 預測！
    df_real_data = ai_engine.generate_backtest_data()

    if df_real_data.empty:
        dbg.error("❌ 無法產生回測資料！")
        exit()

    df_test = df_real_data.tail(test_days)

    print("\n📊 【AI 預測勝率分佈統計】")
    print(df_test['prob_final'].describe())

    dbg.log(f"\n🌟 準備以 {ticker} 過去 {test_days} 天的【純淨未知資料】進行嚴格回測...")
    engine = BacktestEngine(initial_cash=2000000.0)
    engine.run(df_test)
