from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from base import MathTool
from bt.blackboard import Blackboard
from bt.const import BlackboardKey, BTAction, TradeDecision
from bt.core import ActionNode, NodeState
from bt.params import LLMParams, TaxRate
from debug import dbg
from ml.const import OracleCol

if TYPE_CHECKING:
    from ml.model.llm_oracle import GeminiOracle


@dataclass(frozen=True)
class ActionVar:
    TRADE_UNIT: int = 1000

# ==========================================
# 交易動作節點 (虛擬交易執行)
# ==========================================
class ExecuteBuyNode(ActionNode):
    def __init__(self, capital_ratio: float, name: str = BTAction.EXECUTE_BUY):
        super().__init__(name)
        self.capital_ratio = capital_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        # 使用明天的開盤價 (executable_price) 進行扣款
        price = blackboard.executable_price
        # 取得總資產 (包含現金與所有股票市值)
        total_equity = blackboard.account.total_equity
        # 目標曝險金額 = 總資產 * 預設比例 (例如：總資產 100萬 * 0.3 = 目標要買 30萬)
        target_exposure = total_equity * self.capital_ratio
        # 減去已經持有的市值，才是「這次需要買進的預算」
        current_position_value = blackboard.position * blackboard.current_price
        usable_cash = target_exposure - current_position_value

        # 確保不要超過手邊實際還有的現金
        usable_cash = min(usable_cash, blackboard.cash)

        # 防呆：如果算出要買的預算小於 0 (代表已經買超過目標權重了)
        if usable_cash <= 0:
            dbg.war(f"買進取消: 該檔股票當前曝險已達目標比例 ({self.capital_ratio:.0%})")
            return NodeState.FAILURE

        if price <= 0 or pd.isna(price):
            dbg.war(f"買進失敗: 執行價異常 (price={price})")
            return NodeState.FAILURE

        # 漲停板鎖死防禦
        rise_rate = (price - blackboard.current_price) / blackboard.current_price
        if rise_rate >= 0.095:
            dbg.war(f"遭遇漲停板鎖死 (開盤漲幅 {rise_rate:.2%})，無法執行買進！")
            return NodeState.FAILURE

        # 計算應該的稅金
        max_shares_prop = int(usable_cash // (price * (1 + TaxRate.FEE_RATE)))
        max_shares_min = int((usable_cash - TaxRate.MIN_FEE) // price)
        raw_shares = MathTool.clamp(max_shares_prop, 0, max_shares_min)

        vol = blackboard.daily_volume

        max_liquidity_shares = int(vol * 0.05)
        raw_shares = min(raw_shares, max_liquidity_shares)

        # 限制買賣必為整數
        shares_to_buy = (raw_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT

        if shares_to_buy <= 0:
            dbg.war(f"買進失敗: 欲買 0 股。預算={usable_cash:.0f}, 股價={price:.2f}, 流動性上限={max_liquidity_shares}股")
            return NodeState.FAILURE

        # 先計算交易成本，再進行防呆檢查
        raw_cost = shares_to_buy * price
        fee = max(TaxRate.MIN_FEE, raw_cost * TaxRate.FEE_RATE)
        total_cost = raw_cost + fee

        if total_cost > blackboard.cash:
            dbg.war("加計手續費後真實總資金不足！")
            return NodeState.FAILURE

        # 更新黑板帳戶狀態
        old_total_cost = blackboard.position * blackboard.avg_cost
        blackboard.cash -= total_cost

        new_position = blackboard.position + shares_to_buy
        blackboard.avg_cost = (old_total_cost + total_cost) / new_position
        blackboard.position = new_position

        blackboard.action_decision = TradeDecision.BUY

        # 寫入黑板供 Gemini 使用
        blackboard.last_trade_shares = shares_to_buy
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = 0.0

        if blackboard.entry_count == 0:
            blackboard.highest_price = price
        blackboard.entry_count += 1

        dbg.log(f"🟢 [交易執行] 動用 {self.capital_ratio:.0%} 資金買進 {shares_to_buy} 股，成交價 {price:.2f}。剩餘總資金: {blackboard.cash:.0f}")
        blackboard.cached_return_rate = None
        return NodeState.SUCCESS

class ExecuteSellNode(ActionNode):
    def __init__(self, position_ratio: float, name: str = BTAction.EXECUTE_SELL):
        super().__init__(name)
        self.position_ratio = position_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        price = blackboard.executable_price
        position = blackboard.position

        if position <= 0:
            return NodeState.FAILURE

        # 減碼遇到「不足一張」的防呆機制
        if self.position_ratio >= 1.0:
            shares_to_sell = position
        else:
            raw_shares = int(position * self.position_ratio)
            shares_to_sell = (raw_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT

            # 若計算出 0 股，但確實想減碼，則強迫賣出 1 張 (或剩下的全部)
            if shares_to_sell == 0 and raw_shares > 0:
                shares_to_sell = min(position, ActionVar.TRADE_UNIT)

        drop_rate = (price - blackboard.current_price) / blackboard.current_price
        if drop_rate <= -0.095:
            dbg.war(f"遭遇跌停板鎖死 (開盤跌幅 {drop_rate:.2%})，無法執行賣出！")
            return NodeState.FAILURE

        # 流動性限制 (賣出量不得超過當日成交量的 5%)
        max_liquidity_shares = int(blackboard.daily_volume * 0.05)
        if shares_to_sell > max_liquidity_shares:
            dbg.war(f"賣出量 ({shares_to_sell}) 超過市場流動性上限 ({max_liquidity_shares})，僅能部分成交！")
            shares_to_sell = (max_liquidity_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT

        if shares_to_sell <= 0:
            return NodeState.FAILURE

        # 計算賣出實拿金額與損益
        raw_revenue = shares_to_sell * price
        fee = max(TaxRate.MIN_FEE, raw_revenue * TaxRate.FEE_RATE)
        tax = raw_revenue * TaxRate.TAX_RATE
        actual_revenue = raw_revenue - fee - tax

        profit = actual_revenue - (shares_to_sell * blackboard.avg_cost)

        # 更新黑板帳戶狀態
        blackboard.cash += actual_revenue
        blackboard.position -= shares_to_sell

        if blackboard.position == 0:
            # 已經清倉：一鍵清除所有記憶
            blackboard.clear_trade_memory()
        elif self.position_ratio < 1.0:
            # 部分減碼：標記已經停利過，防止碎肉機陷阱
            blackboard.is_partial_profit_taken = True

        blackboard.action_decision = TradeDecision.SELL
        blackboard.last_trade_shares = shares_to_sell
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = profit

        dbg.log(f"[交易執行] 賣出 {self.position_ratio:.0%} 部位 ({shares_to_sell} 股)，成交價 {price:.2f}。淨損益: {profit:.2f}。目前資金: {blackboard.cash:.2f}")
        blackboard.cached_return_rate = None
        return NodeState.SUCCESS

class ExecuteHoldNode(ActionNode):
    """保持觀望，不進行任何交易"""
    def __init__(self, name: str = BTAction.EXECUTE_HOLD):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        blackboard.action_decision = TradeDecision.HOLD
        dbg.log("[交易執行] 維持現狀 (HOLD)。")
        return NodeState.SUCCESS


# ==========================================
# AI 報告生成節點 (呼叫 Gemini)
# ==========================================
class GenerateGeminiReportNode(ActionNode):
    """
    負責收集黑板上的所有資訊，打包成 Prompt，並呼叫 Gemini 產出分析報告。
    內建「防幻覺與嚴禁竄改」的 System Prompt 框架。
    """
    def __init__(self, oracle=None, name: str = BTAction.GENERATE_REPORT):
        super().__init__(name)
        self.oracle = oracle

    def tick(self, blackboard: Blackboard) -> NodeState:
        dbg.log("正在打包決策脈絡，準備生成 AI 覆盤報告...")

        active_oracle: "GeminiOracle" | None = getattr(blackboard, BlackboardKey.ORACLE.value, self.oracle)

        if not active_oracle:
            dbg.war("⚠️ [Debug 警告] 找不到 Gemini Oracle 實體！將強制降級為『模擬報告』。請確認 API Key 是否載入，且有綁定至 Blackboard。")
        else:
            dbg.log("✅ [Debug 確認] 成功偵測到 Gemini Oracle 實體，準備呼叫真實 AI 模型！")

        action_val = blackboard.action_decision

        trade_info_str = ""
        if action_val == TradeDecision.BUY:
            trade_info_str = f"- 實際執行動作：系統已成功【買進】 {blackboard.last_trade_shares} 股，建議掛單價 {blackboard.last_trade_price:.2f} 元。"
        elif action_val == TradeDecision.SELL:
            action_type = "全數出清" if blackboard.position == 0 else "部分減碼"
            trade_info_str = f"- 實際執行動作：系統已成功【{action_type}】 {blackboard.last_trade_shares} 股，建議掛單價 {blackboard.last_trade_price:.2f} 元。"
        elif action_val == TradeDecision.HOLD:
            trade_info_str = "- 實際執行動作：系統判定維持現狀【觀望 (HOLD)】。"
        else:
            trade_info_str = f"- 實際執行動作：未知狀態 ({action_val})。"

        score = getattr(blackboard, OracleCol.SCORE.value, LLMParams.DEFAULT_SENTIMENT_SCORE)
        reason = getattr(blackboard, OracleCol.REASON.value, '無相關新聞或未啟用 LLM')

        pricing_logic = getattr(blackboard, 'gemini_reasoning', '')

        action_upper = str(action_val).upper()
        prompt = f"""
        【最高指令】：你是一個只負責「事後覆盤」的量化分析助理。
        系統【已經】做出了最終交易決策，你絕對不可以質疑、修改或建議更改該決策。你的唯一任務是根據以下數據，寫出一份 100 字左右的專業、客觀的繁體中文盤後報告。

        【當前決策事實 (不可竄改)】
        - 股票代號：{blackboard.ticker}
        - 系統最終決策：{action_upper}
        {trade_info_str}
        - 目前總持股：{blackboard.position} 股 (剩餘現金：{blackboard.cash:.2f} 元)

        【AI 模型內部視角】
        - 總指揮 (Meta-Learner) 綜合勝率：{blackboard.prob_final:.2%}
        - 左腦 (XGBoost 技術面)：{blackboard.prob_xgb:.2%}
        - 右腦 (DL 深度學習 K 線)：{blackboard.prob_dl:.2%}
        - 第三腦 (Market Brain) 安全度：{getattr(blackboard, 'prob_market_safe', 1.0):.2%}
        - LLM 新聞情緒：{score} 分 (1-10分)。判讀理由：{reason}

        【演算法智慧定價考量】
        {pricing_logic}

        【報告撰寫指引】
        1. 破題直接說明系統今天執行了什麼動作 ({action_upper})。
        2. 綜合技術面勝率、大盤雷達、新聞情緒，以及「智慧定價的掛單考量」，簡述「為什麼系統會觸發這個動作與定價」。
        3. 語氣保持冷靜、客觀的法人機構風格，不使用強烈情緒化字眼。
        """

        try:
            if active_oracle:
                final_report = active_oracle.generate_report(prompt)
            else:
                final_report = f"【模擬 AI 覆盤報告】\n系統今日對 {blackboard.ticker} 執行 {action_upper}。主要驅動力來自綜合勝率達 {blackboard.prob_final:.2%}，且大盤防禦雷達顯示環境安全。儘管新聞情緒呈現 {score} 分 ({reason})，系統仍依紀律執行既定策略..."

            if pricing_logic:
                blackboard.gemini_reasoning = f"{final_report}\n{pricing_logic}"
            else:
                blackboard.gemini_reasoning = final_report

            dbg.log(f"📝 報告生成完畢：\n{final_report}")
            return NodeState.SUCCESS

        except Exception as e:
            dbg.error(f"Gemini 報告生成節點發生例外: {e}")
            blackboard.gemini_reasoning = "模組發生例外，無法生成真實報告。"
            return NodeState.FAILURE