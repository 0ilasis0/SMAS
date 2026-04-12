from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bt.account import Account, Position
from bt.blackboard import Blackboard
from bt.const import BlackboardKey, TradeDecision
from bt.strategy import build_trading_tree
from bt.strategy_config import StrategyConfig
from const import Color
from data.const import StockCol
from debug import dbg
from ml.const import FeatureCol, SignalCol
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
    # 績效分析相關
    PEAK = "Peak"
    DRAWDOWN = "Drawdown"
    DAILY_RETURN = "Daily_Return"

@dataclass(frozen=True)
class BacktestRecord:
    """單日回測數據載體"""
    Date: datetime
    Close: float
    Cash: float
    Position: int
    Total_Equity: float
    Action: TradeDecision
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
    def __init__(self, initial_cash: int, ticker: str, strategy: StrategyConfig =  StrategyConfig()):
        self.initial_cash = initial_cash
        self.ticker = ticker
        self.account = Account(cash=initial_cash)
        # 初始化時，為這檔股票建立一個空的持倉紀錄
        self.account.positions[self.ticker] = Position()

        self.bb = Blackboard(ticker=ticker, account=self.account)

        self.tree = build_trading_tree(strategy)

        self.history_records = []

    def run(self, df: pd.DataFrame, silence: bool = False):
        """
        執行回測
        """
        with dbg.silence(active=silence):
            self.history_records.clear()

            # 將帳戶與黑板狀態重置為初始狀態
            self.account.cash = self.initial_cash
            self.account.positions[self.ticker] = Position()
            self.bb.clear_trade_memory()

            dbg.log(f"🚀 開始執行行為樹回測，初始資金: {self.initial_cash:,.0f} 元，共 {len(df)} 個交易日...")

            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i + 1]

                date = df.index[i]
                current_close = row[StockCol.CLOSE.value]

                # 確保帳戶知道最新的收盤價，以計算真實的 total_equity
                self.account.update_price(self.ticker, current_close)

                # 將今天的收盤資訊與明天的「開盤價」、「成交量」傳給黑板
                self.bb.current_date = str(date)

                self.bb.update_price(
                    current_price=current_close,
                    high_price=row[StockCol.HIGH.value],
                    executable_price=next_row[StockCol.OPEN.value],  # 實際執行交易的價格
                    daily_volume=next_row[StockCol.VOLUME.value]     # 流動性上限
                )

                self.bb.prob_market_safe = row.get(SignalCol.PROB_MARKET_SAFE.value, 1.0)
                self.bb.prob_final = row[SignalCol.PROB_FINAL.value]
                self.bb.prob_xgb = row[SignalCol.PROB_XGB.value]
                self.bb.prob_dl = row[SignalCol.PROB_DL.value]

                self.bb.bias_20 = row.get(FeatureCol.BIAS_MONTH.value, 0.0)
                self.bb.return_5d = row.get(FeatureCol.RETURN_5D.value, 0.0)

                # 清空前一天的決策紀錄
                self.bb.action_decision = TradeDecision.HOLD

                # 全域時鐘，每天確實扣減冷卻期
                current_cd = getattr(self.bb, BlackboardKey.COOLDOWN_TIMER.value, 0)
                if current_cd > 0:
                    setattr(self.bb, BlackboardKey.COOLDOWN_TIMER.value, current_cd - 1)

                # 執行行為樹心跳 (Tick)
                self.tree.tick(self.bb)

                # 計算當日總淨值
                stock_value = self.bb.position * current_close
                total_equity = self.bb.cash + stock_value

                # 確保動作執行後，將黑板的持倉狀態同步回 Account
                # (雖然在 action.py 裡面你可能是扣 account.cash，但 position 的同步非常重要)
                self.account.positions[self.ticker].shares = self.bb.position
                self.account.positions[self.ticker].avg_cost = self.bb.avg_cost

                total_equity = self.account.total_equity

                # 紀錄歷史
                record = BacktestRecord(
                    Date=date,
                    Close=current_close,
                    Cash=self.account.cash,
                    Position=self.bb.position,
                    Total_Equity=total_equity,
                    Action=self.bb.action_decision,
                    prob_final=self.bb.prob_final,
                    prob_market_safe=self.bb.prob_market_safe
                )
                self.history_records.append(record.to_dict())

            if not df.empty:
                last_date = df.index[-1]
                last_row = df.iloc[-1]
                last_close = last_row[StockCol.CLOSE.value]

                self.account.update_price(self.ticker, last_close)
                last_equity = self.account.total_equity

                final_record = BacktestRecord(
                    Date=last_date,
                    Close=last_close,
                    Cash=self.account.cash,
                    Position=self.bb.position,
                    Total_Equity=last_equity,
                    Action=TradeDecision.HOLD, # 最後一天不動作
                    prob_final=last_row.get(SignalCol.PROB_FINAL.value, 0.5),
                    prob_market_safe=last_row.get(SignalCol.PROB_MARKET_SAFE.value, 1.0)
                )
                self.history_records.append(final_record.to_dict())

            report_stats = self._generate_report(disable_plot=silence)

            return report_stats

    def _generate_report(self, disable_plot: bool):
        """計算績效指標並繪製三層量化儀表板"""
        if not self.history_records: return {}

        df_res = pd.DataFrame(self.history_records).set_index(HistoryCol.DATE)

        final_equity = df_res[HistoryCol.TOTAL_EQUITY].iloc[-1]
        total_return = (final_equity - self.initial_cash) / self.initial_cash

        # 計算年化報酬率 (CAGR)
        trading_days = len(df_res)
        cagr = (final_equity / self.initial_cash) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 計算 MDD
        df_res[HistoryCol.PEAK] = df_res[HistoryCol.TOTAL_EQUITY].cummax()
        df_res[HistoryCol.DRAWDOWN] = (df_res[HistoryCol.TOTAL_EQUITY] - df_res[HistoryCol.PEAK]) / df_res[HistoryCol.PEAK]
        max_drawdown = df_res[HistoryCol.DRAWDOWN].min()

        # 計算夏普值 (Sharpe Ratio)
        df_res[HistoryCol.DAILY_RETURN] = df_res[HistoryCol.TOTAL_EQUITY].pct_change().fillna(0)
        daily_volatility = df_res[HistoryCol.DAILY_RETURN].std()
        sharpe_ratio = (df_res[HistoryCol.DAILY_RETURN].mean() - (0.01 / 252)) / daily_volatility * np.sqrt(252) if daily_volatility > 0 else 0.0

        buy_count = len(df_res[df_res[HistoryCol.ACTION] == TradeDecision.BUY])
        sell_count = len(df_res[df_res[HistoryCol.ACTION] == TradeDecision.SELL])

        if not disable_plot:
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

            fig = plt.figure(figsize=(14, 10))

            # 第一層：資金曲線與回撤
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(df_res.index, df_res[HistoryCol.TOTAL_EQUITY], label='Total Equity', color=Color.BLUE, linewidth=2)
            ax1.set_title(f'IDSS Quant Strategy Dashboard - {self.bb.ticker}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Total Equity (NTD)')
            ax1.grid(True, alpha=0.3)

            ax1_dd = ax1.twinx()
            ax1_dd.fill_between(df_res.index, df_res[HistoryCol.DRAWDOWN], 0, color=Color.RED, alpha=0.2, label=HistoryCol.DRAWDOWN)
            ax1_dd.set_ylabel(f'{HistoryCol.DRAWDOWN} (%)', color=Color.RED)
            ax1_dd.tick_params(axis='y', labelcolor=Color.RED)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax1_dd.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

            # 第二層：股價走勢與買賣點
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(df_res.index, df_res[HistoryCol.CLOSE], label='Stock Price', color=Color.GRAY, alpha=0.7)
            buys = df_res[df_res[HistoryCol.ACTION] == TradeDecision.BUY]
            sells = df_res[df_res[HistoryCol.ACTION] == TradeDecision.SELL]
            ax2.scatter(buys.index, buys[HistoryCol.CLOSE], marker='^', color=Color.GREEN, s=100, label='Buy', zorder=5)
            ax2.scatter(sells.index, sells[HistoryCol.CLOSE], marker='v', color=Color.RED, s=100, label='Sell', zorder=5)
            ax2.set_ylabel('Stock Price')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')

            # 第三層：AI 勝率與大盤防禦雷達
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(df_res.index, df_res[HistoryCol.PROB_FINAL], label='AI Final Prob', color=Color.ORANGE, linewidth=1.5)
            ax3.plot(df_res.index, df_res[HistoryCol.PROB_MARKET], label='Market Safety Prob', color=Color.PURPLE, linestyle='--', linewidth=1.5)
            ax3.axhline(y=0.5, color=Color.RED, linestyle=':', alpha=0.5, label='50% Threshold')

            ax3.fill_between(
                df_res.index, 0, 1,
                where=(df_res[HistoryCol.PROB_MARKET] < 0.5),
                color=Color.RED, alpha=0.1, label='Market Danger Zone', transform=ax3.get_xaxis_transform()
            )

            ax3.set_ylabel('Probability')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')

            plt.tight_layout()

            # --- 若之後要存檔，可以解開註解 ---
            # try:
            #     report_img_path = PathConfig.get_chart_report_path(ticker=self.bb.ticker)
            #     report_img_path.parent.mkdir(parents=True, exist_ok=True)
            #     fig.savefig(str(report_img_path), dpi=300, bbox_inches='tight')
            # except Exception as e:
            #     dbg.error(f"圖片存檔失敗: {e}")

            # 關閉畫布釋放記憶體
            plt.close(fig)

        # 無論有沒有畫圖，都回傳純數字字典
        stats = {
            "initial_cash": self.initial_cash,
            "final_equity": final_equity,
            "total_return": total_return,
            "cagr": cagr,
            "mdd": max_drawdown,
            "sharpe": sharpe_ratio,
            "buy_count": buy_count,
            "sell_count": sell_count
        }

        return stats
