from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum

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
from path import PathConfig


class HistoryCol(StrEnum):
    """回測紀錄欄位名稱常量"""
    DATE = "Date"
    CLOSE = "Close"
    CASH = "Cash"
    POSITION = "Position"
    TOTAL_EQUITY = "Total_Equity"
    ACTION = "Action"
    PROB_FINAL = "prob_final"
    PROB_MARKET = "prob_market_safe"

@dataclass(frozen=True)
class BacktestRecord:
    """單日回測數據載體"""
    Date: datetime
    Close: float
    Cash: float
    Position: int
    Total_Equity: float
    Action: DecisionAction
    prob_final: float
    prob_market_safe: float

    def to_dict(self):
        """轉換為字典供 Pandas 使用"""
        return asdict(self)

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
            self.bb.prob_market_safe = row.get(MetaCol.PROB_MARKET_SAFE, 1.0)
            self.bb.prob_final = row[MetaCol.PROB_FINAL]

            self.bb.prob_xgb = row[MetaCol.PROB_XGB]
            self.bb.prob_dl = row[MetaCol.PROB_DL]

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
            record = BacktestRecord(
                Date=date,
                Close=current_close,
                Cash=self.bb.cash,
                Position=self.bb.position,
                Total_Equity=total_equity,
                Action=self.bb.action_decision,
                prob_final=self.bb.prob_final,
                prob_market_safe=self.bb.prob_market_safe
            )
            self.history_records.append(record.to_dict())

        if len(df) > 0:
            last_date = df.index[-1]
            last_row = df.iloc[-1]
            last_close = last_row[StockCol.CLOSE]
            last_equity = self.bb.cash + (self.bb.position * last_close)

            final_record = BacktestRecord(
                Date=last_date,
                Close=last_close,
                Cash=self.bb.cash,
                Position=self.bb.position,
                Total_Equity=last_equity,
                Action=DecisionAction.HOLD, # 最後一天不動作
                prob_final=last_row.get(MetaCol.PROB_FINAL, 0.5),
                prob_market_safe=last_row.get(MetaCol.PROB_MARKET_SAFE, 1.0)
            )
            self.history_records.append(final_record.to_dict())

        self._generate_report()

    def _generate_report(self):
        """計算績效指標並繪製三層量化儀表板"""
        if not self.history_records:
            return

        df_res = pd.DataFrame(self.history_records).set_index(HistoryCol.DATE)

        final_equity = df_res[HistoryCol.TOTAL_EQUITY].iloc[-1]
        total_return = (final_equity - self.initial_cash) / self.initial_cash

        # 計算年化報酬率 (CAGR)
        trading_days = len(df_res)
        cagr = (final_equity / self.initial_cash) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 計算 MDD
        df_res['Peak'] = df_res[HistoryCol.TOTAL_EQUITY].cummax()
        df_res['Drawdown'] = (df_res[HistoryCol.TOTAL_EQUITY] - df_res['Peak']) / df_res['Peak']
        max_drawdown = df_res['Drawdown'].min()

        # 計算夏普值 (Sharpe Ratio)
        df_res['Daily_Return'] = df_res[HistoryCol.TOTAL_EQUITY].pct_change().fillna(0)
        daily_volatility = df_res['Daily_Return'].std()
        sharpe_ratio = (df_res['Daily_Return'].mean() - (0.01 / 252)) / daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0.0

        # 統計交易次數
        buy_count = len(df_res[df_res[HistoryCol.ACTION] == DecisionAction.BUY])
        sell_count = len(df_res[df_res[HistoryCol.ACTION] == DecisionAction.SELL])

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

        # 整合為一個專業的三層儀表板
        plt.figure(figsize=(14, 10))

        # 第一層：資金曲線與回撤
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df_res.index, df_res[HistoryCol.TOTAL_EQUITY], label='Total Equity', color='blue', linewidth=2)
        ax1.set_title(f'IDSS Quant Strategy Dashboard - {self.bb.ticker}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Equity (NTD)')
        ax1.grid(True, alpha=0.3)

        # 在同一個圖表加入回撤 (Drawdown) 的紅色面積圖，共用 X 軸但使用右側 Y 軸
        ax1_dd = ax1.twinx()
        ax1_dd.fill_between(df_res.index, df_res['Drawdown'], 0, color='red', alpha=0.2, label='Drawdown')
        ax1_dd.set_ylabel('Drawdown (%)', color='red')
        ax1_dd.tick_params(axis='y', labelcolor='red')

        # 合併兩個軸的圖例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_dd.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # 第二層：股價走勢與買賣點
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(df_res.index, df_res[HistoryCol.CLOSE], label='Stock Price', color='gray', alpha=0.7)
        buys = df_res[df_res[HistoryCol.ACTION] == DecisionAction.BUY]
        sells = df_res[df_res[HistoryCol.ACTION] == DecisionAction.SELL]
        ax2.scatter(buys.index, buys[HistoryCol.CLOSE], marker='^', color='green', s=100, label='Buy', zorder=5)
        ax2.scatter(sells.index, sells[HistoryCol.CLOSE], marker='v', color='red', s=100, label='Sell', zorder=5)
        ax2.set_ylabel('Stock Price')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # 第三層：AI 勝率與大盤防禦雷達
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(df_res.index, df_res[HistoryCol.PROB_FINAL], label='AI Final Prob', color='orange', linewidth=1.5)
        ax3.plot(df_res.index, df_res[HistoryCol.PROB_MARKET], label='Market Safety Prob', color='purple', linestyle='--', linewidth=1.5)
        ax3.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='50% Threshold')

        # 將大盤危險區域標示為紅色背景
        ax3.fill_between(
            df_res.index, 0, 1,
            where=(df_res[HistoryCol.PROB_MARKET] < 0.5),
            color='red', alpha=0.1, label='Market Danger Zone', transform=ax3.get_xaxis_transform()
        )

        ax3.set_ylabel('Probability')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')

        plt.tight_layout()

        try:
            report_img_path = PathConfig.get_chart_report_path(ticker=ticker)
            report_img_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(report_img_path, dpi=300)
            dbg.log(f"📸 儀表板圖片已自動儲存至: {report_img_path}")
        except Exception as e:
            dbg.war(f"圖片存檔失敗: {e}")

        plt.show()


if __name__ == "__main__":
    from ml.engine import QuantAIEngine

    ticker = "2388.TW"
    test_days = 240
    ai_engine = QuantAIEngine(ticker=ticker, oos_days=test_days)

    # 假設你需要重新上網爬資料 (如果已經有資料了，這段可以註解)
    ai_engine.update_market_data()
    # 假設你需要重新訓練模型 (如果模型已經是乾淨的，這段可以註解)
    ai_engine.train_all_models(save_models=True)

    if not ai_engine.load_inference_models():
        dbg.error("❌ 模型載入失敗...")
        exit()

    df_real_data = ai_engine.generate_backtest_data()

    if df_real_data.empty:
        dbg.error("❌ 無法產生回測資料！")
        exit()

    df_test = df_real_data.tail(test_days)

    print("\n📊 【AI 預測勝率分佈統計】")
    print(df_test[MetaCol.PROB_FINAL].describe())

    dbg.log(f"\n🌟 準備以 {ticker} 過去 {test_days} 天的【純淨未知資料】進行嚴格回測...")
    engine = BacktestEngine(initial_cash=2000000)
    engine.run(df_test)
