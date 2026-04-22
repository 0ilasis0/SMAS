from dataclasses import dataclass

import pandas as pd

from base import MathTool
from bt.blackboard import Blackboard
from bt.const import BlackboardKey, BTAction, TradeDecision
from bt.core import ActionNode, NodeState
from bt.params import TaxRate
from debug import dbg


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
        price = blackboard.executable_price
        total_equity = blackboard.account.total_equity
        target_exposure = total_equity * self.capital_ratio
        current_position_value = blackboard.position * blackboard.current_price
        usable_cash = target_exposure - current_position_value

        # 將 blackboard.cash 改為存取新版的 blackboard.account.total_cash
        usable_cash = min(usable_cash, blackboard.account.total_cash)

        if usable_cash <= 0:
            dbg.war(f"買進取消: 該檔股票當前曝險已達目標比例 ({self.capital_ratio:.0%})")
            return NodeState.FAILURE

        if price <= 0 or pd.isna(price):
            dbg.war(f"買進失敗: 執行價異常 (price={price})")
            return NodeState.FAILURE

        rise_rate = (price - blackboard.current_price) / blackboard.current_price
        if rise_rate >= 0.095:
            dbg.war(f"遭遇漲停板鎖死 (開盤漲幅 {rise_rate:.2%})，無法執行買進！")
            return NodeState.FAILURE

        max_shares_prop = int(usable_cash // (price * (1 + TaxRate.FEE_RATE)))
        max_shares_min = int((usable_cash - TaxRate.MIN_FEE) // price)
        raw_shares = MathTool.clamp(max_shares_prop, 0, max_shares_min)

        vol = blackboard.daily_volume
        max_liquidity_shares = int(vol * 0.05)
        raw_shares = min(raw_shares, max_liquidity_shares)

        shares_to_buy = (raw_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT

        if shares_to_buy <= 0:
            dbg.war(f"買進失敗: 欲買 0 股。預算={usable_cash:.0f}, 股價={price:.2f}, 流動性上限={max_liquidity_shares}股")
            return NodeState.FAILURE

        raw_cost = shares_to_buy * price
        fee = max(TaxRate.MIN_FEE, raw_cost * TaxRate.FEE_RATE)
        total_cost = raw_cost + fee

        # 改用 blackboard.account.total_cash
        if total_cost > blackboard.account.total_cash:
            dbg.war("加計手續費後真實總資金不足！")
            return NodeState.FAILURE

        old_total_cost = blackboard.position * blackboard.avg_cost

        # 扣除現金
        blackboard.account.total_cash -= total_cost

        new_position = blackboard.position + shares_to_buy
        blackboard.avg_cost = (old_total_cost + total_cost) / new_position
        blackboard.position = new_position

        blackboard.action_decision = TradeDecision.BUY

        blackboard.last_trade_shares = shares_to_buy
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = 0.0

        if blackboard.entry_count == 0:
            blackboard.highest_price = price
        blackboard.entry_count += 1

        dbg.log(f"🟢 [交易執行] 動用 {self.capital_ratio:.0%} 資金買進 {shares_to_buy} 股，成交價 {price:.2f}。剩餘總資金: {blackboard.account.total_cash:.0f}")
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

        if self.position_ratio >= 1.0:
            shares_to_sell = position
        else:
            raw_shares = int(position * self.position_ratio)
            shares_to_sell = (raw_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT
            if shares_to_sell == 0 and raw_shares > 0:
                shares_to_sell = min(position, ActionVar.TRADE_UNIT)

        drop_rate = (price - blackboard.current_price) / blackboard.current_price
        if drop_rate <= -0.095:
            dbg.war(f"遭遇跌停板鎖死 (開盤跌幅 {drop_rate:.2%})，無法執行賣出！")
            return NodeState.FAILURE

        max_liquidity_shares = int(blackboard.daily_volume * 0.05)
        if shares_to_sell > max_liquidity_shares:
            dbg.war(f"賣出量 ({shares_to_sell}) 超過市場流動性上限 ({max_liquidity_shares})，僅能部分成交！")
            shares_to_sell = (max_liquidity_shares // ActionVar.TRADE_UNIT) * ActionVar.TRADE_UNIT

        if shares_to_sell <= 0:
            return NodeState.FAILURE

        raw_revenue = shares_to_sell * price
        fee = max(TaxRate.MIN_FEE, raw_revenue * TaxRate.FEE_RATE)
        tax = raw_revenue * TaxRate.TAX_RATE
        actual_revenue = raw_revenue - fee - tax
        profit = actual_revenue - (shares_to_sell * blackboard.avg_cost)

        # 賣出獲得的現金，加回 blackboard.account.total_cash
        blackboard.account.total_cash += actual_revenue
        blackboard.position -= shares_to_sell

        if blackboard.position == 0:
            blackboard.clear_trade_memory()
        elif self.position_ratio < 1.0:
            blackboard.is_partial_profit_taken = True

        blackboard.action_decision = TradeDecision.SELL
        blackboard.last_trade_shares = shares_to_sell
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = profit

        dbg.log(f"[交易執行] 賣出 {self.position_ratio:.0%} 部位 ({shares_to_sell} 股)，成交價 {price:.2f}。淨損益: {profit:.2f}。目前資金: {blackboard.account.total_cash:.2f}")
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

    def tick(self, bb: "Blackboard") -> NodeState:
        # 1. 確認是否有啟用 LLM 引擎
        oracle = self.oracle or bb.get(BlackboardKey.ORACLE.value)
        if not oracle:
            dbg.war(f"[{bb.ticker}] 未啟用 LLM Oracle，跳過戰報生成，保留系統原始日誌。")
            return NodeState.SUCCESS

        dbg.log(f"[{bb.ticker}] 正在將決策資料封裝送往 Gemini 撰寫專業戰報...")

        # ==========================================
        # 步驟 A：建構 Control Plane (系統最高權限指令)
        # ==========================================
        # 這是寫死在後端、絕對不可被竄改的「大腦潛意識」
        base_system_prompt = (
            "你是一位頂尖的量化基金經理人與市場分析師。\n"
            "你的任務是根據我（系統）提供的【量化決策黑板資料】，撰寫一份專業、簡潔、具備行動力的投資戰報。\n"
            "【嚴格紀律與防幻覺規則】：\n"
            "1. 你絕對不可竄改、懷疑或反駁系統給出的「最終決策 (BUY/SELL/HOLD)」、「預期掛單價」與「機率數字」。\n"
            "2. 你的工作是『潤飾與解釋』這個決策，讓一般投資人能看懂，而不是『重新決策』。\n"
            "3. 語氣要果斷、冷靜、充滿華爾街數據感，拒絕模稜兩可的廢話與投資警語（系統已有免責聲明）。"
        )

        # 從黑板提取動態塞入的「總裁指令」(例如：洗盤警告、法說會避險)
        dynamic_directives = "\n".join(bb.system_directives) if hasattr(bb, 'system_directives') else ""

        final_system_instruction = base_system_prompt
        if dynamic_directives:
            final_system_instruction += f"\n\n【🚨 總裁特級動態指令 (最高執行權限)】：\n{dynamic_directives}"

        # ==========================================
        # 步驟 B：建構 Data Plane (供 LLM 閱讀的特徵資料)
        # ==========================================
        # 這是送進去讓 LLM "閱讀" 的客觀資料，它不能違背上面 System Prompt 的規範
        user_prompt = f"""
        【系統提示】
        無須列出戰報日期時間

        請根據以下系統輸出的資料，生成今日戰報：

        【標的資訊】
        - 股票代號：{bb.ticker}
        - 今日參考價：{getattr(bb, 'current_price', '未知')}
        - 20 日月乖離率：{getattr(bb, 'bias_20', 0):.2%}

        【最終戰術決策】
        - 行動指令：{getattr(bb, 'action_decision', '未知')}
        - 系統建議掛單價：{getattr(bb, 'last_trade_price', '未知')}

        【AI 雙腦預測勝率】
        - 綜合終極勝率：{getattr(bb, 'prob_final', 0):.0%}
        - XGBoost 動能預測勝率：{getattr(bb, 'prob_xgb', 0):.0%}
        - LSTM 序列預測勝率：{getattr(bb, 'prob_dl', 0):.0%}
        - 大盤環境安全度：{getattr(bb, 'prob_market_safe', 0):.0%}

        【量化系統邏輯推演理由 (原始輸出，請潤飾此段)】
        {getattr(bb, 'gemini_reasoning', '無特別理由')}
        """

        # ==========================================
        # 步驟 C：呼叫 Oracle API 並寫回黑板
        # ==========================================
        try:
            report_text = oracle.generate_report(
                system_instruction=final_system_instruction,
                user_prompt=user_prompt
            )

            # 將 Gemini 寫好的優美戰報，覆寫回黑板原本冷硬的 gemini_reasoning 欄位，供 UI 讀取
            bb.gemini_reasoning = report_text

        except Exception as e:
            dbg.error(f"Gemini 戰報生成失敗: {e}")
            bb.gemini_reasoning = f"【AI 戰報生成失敗，顯示系統原始邏輯】\n\n{bb.gemini_reasoning}"

        return NodeState.SUCCESS