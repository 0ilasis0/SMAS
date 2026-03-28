import pandas as pd

from bt.blackboard import Blackboard
from bt.const import BtVar, DecisionAction, ExecuteCol
from bt.core import BaseNode, NodeState
from bt.params import TaxRate
from debug import dbg


# 交易動作節點 (虛擬交易執行)
class ExecuteBuyNode(BaseNode):
    def __init__(self, capital_ratio: float, name: str = ExecuteCol.BUY):
        super().__init__(name)
        self.capital_ratio = capital_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        # 使用明天的開盤價 (executable_price) 進行扣款
        price = blackboard.executable_price
        usable_cash = blackboard.cash * self.capital_ratio

        if price <= 0 or pd.isna(price):
            dbg.war(f"買進失敗: 執行價異常 (price={price})")
            return NodeState.FAILURE

        # 漲停板鎖死防禦
        rise_rate = (price - blackboard.current_price) / blackboard.current_price
        if rise_rate >= 0.095:
            dbg.war(f"遭遇漲停板鎖死 (開盤漲幅 {rise_rate:.2%})，無法執行買進！")
            return NodeState.FAILURE

        max_shares_prop = int(usable_cash // (price * (1 + TaxRate.FEE_RATE)))
        max_shares_min = int((usable_cash - TaxRate.MIN_FEE) // price)
        raw_shares = max(0, min(max_shares_prop, max_shares_min))

        # 恢復買進流動性限制 (買進量不得超過當日成交量的 5%)
        max_liquidity_shares = int(blackboard.daily_volume * 0.05)
        raw_shares = min(raw_shares, max_liquidity_shares)

        # 嚴格限制整股
        shares_to_buy = (raw_shares // BtVar.TRADE_UNIT) * BtVar.TRADE_UNIT
        if shares_to_buy <= 0:
            dbg.war(f"買進失敗: 欲買 {shares_to_buy} 股。可用資金={usable_cash:.0f}, 股價={price:.2f}, 原始試算={raw_shares} 股")
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
        blackboard.action_decision = DecisionAction.BUY

        # 寫入黑板供 Gemini 使用
        blackboard.last_trade_shares = shares_to_buy
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = 0.0

        if blackboard.entry_count == 0:
            blackboard.highest_price = price
        blackboard.entry_count += 1

        dbg.log(f"🟢 [交易執行] 動用 {self.capital_ratio:.0%} 資金買進 {shares_to_buy} 股，成交價 {price:.2f}。剩餘總資金: {blackboard.cash:.2f}")
        blackboard.cached_return_rate = None
        return NodeState.SUCCESS


class ExecuteSellNode(BaseNode):
    def __init__(self, position_ratio: float, name: str = ExecuteCol.SELL):
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
            shares_to_sell = (raw_shares // BtVar.TRADE_UNIT) * BtVar.TRADE_UNIT

            # 若計算出 0 股，但確實想減碼，則強迫賣出 1 張 (或剩下的全部)
            if shares_to_sell == 0 and raw_shares > 0:
                shares_to_sell = min(position, BtVar.TRADE_UNIT)


        drop_rate = (price - blackboard.current_price) / blackboard.current_price
        if drop_rate <= -0.095:
            dbg.war(f"遭遇跌停板鎖死 (開盤跌幅 {drop_rate:.2%})，無法執行賣出！")
            return NodeState.FAILURE

        # 🚀 升級：流動性限制 (賣出量不得超過當日成交量的 5%)
        max_liquidity_shares = int(blackboard.daily_volume * 0.05)
        if shares_to_sell > max_liquidity_shares:
            dbg.war(f"賣出量 ({shares_to_sell}) 超過市場流動性上限 ({max_liquidity_shares})，僅能部分成交！")
            shares_to_sell = (max_liquidity_shares // BtVar.TRADE_UNIT) * BtVar.TRADE_UNIT

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

        blackboard.action_decision = DecisionAction.SELL
        blackboard.last_trade_shares = shares_to_sell
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = profit

        dbg.log(f"[交易執行] 賣出 {self.position_ratio:.0%} 部位 ({shares_to_sell} 股)，成交價 {price:.2f}。淨損益: {profit:.2f}。目前資金: {blackboard.cash:.2f}")
        blackboard.cached_return_rate = None
        return NodeState.SUCCESS

class ExecuteHoldNode(BaseNode):
    """保持觀望，不進行任何交易"""
    def __init__(self, name: str = ExecuteCol.HOLD):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        blackboard.action_decision = DecisionAction.HOLD
        dbg.log("[交易執行] 維持現狀 (HOLD)。")
        return NodeState.SUCCESS


class IgnoreFailure(BaseNode):
    """裝飾節點：將子節點的 FAILURE 強制轉為 SUCCESS (非關鍵路徑避震器)"""
    def __init__(self, child: BaseNode):
        super().__init__(child.name + "_Ignored")
        self.child = child

    def tick(self, blackboard: Blackboard) -> NodeState:
        state = self.child.tick(blackboard)
        # 如果子節點還在 RUNNING，就乖乖向上回傳 RUNNING (未來擴充非同步 API 時會用到)
        if state == NodeState.RUNNING:
            return NodeState.RUNNING
        return NodeState.SUCCESS


# AI 報告生成節點 (呼叫 Gemini)
class GenerateGeminiReportNode(BaseNode):
    """
    負責收集黑板上的所有資訊，打包成 Prompt，並呼叫 Gemini 產出分析報告。
    """
    def __init__(self, name: str = BtVar.GENERATE_GEMINI_REPORT):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        dbg.log("正在呼叫 Gemini 生成決策分析報告...")

        trade_info_str = ""
        if blackboard.action_decision == DecisionAction.BUY:
            trade_info_str = f"- 交易執行：系統買進了 {blackboard.last_trade_shares} 股，成交價 {blackboard.last_trade_price:.2f}。"
        elif  blackboard.action_decision == DecisionAction.SELL:
            action_type = "全數出清" if blackboard.position == 0 else "部分減碼"
            trade_info_str = f"- 交易執行：系統{action_type} {blackboard.last_trade_shares} 股，成交價 {blackboard.last_trade_price:.2f}，本次交易淨損益為 {blackboard.last_trade_profit:.2f} 元。"
        else:
            trade_info_str = "- 交易執行：系統判定無顯著訊號或條件不符，維持空手/持倉不動 (HOLD)。"

        prompt = f"""
        你是一位專業的台灣股票量化交易分析師。請根據以下的 AI 模型預測結果，撰寫一份簡短且專業的繁體中文分析報告。

        【標的與決策】
        - 股票代號：{blackboard.ticker}
        - 系統最終決策：{blackboard.action_decision}
        {trade_info_str}

        【當前帳戶狀態】
        - 目前持有股數：{blackboard.position} 股
        - 帳上剩餘現金：{blackboard.cash:.2f} 元

        【模型勝率數據】
        - 左腦 (XGBoost 技術指標) 預測上漲機率：{blackboard.prob_xgb:.2%}
        - 右腦 (CNN-RNN K線型態) 預測上漲機率：{blackboard.prob_dl:.2%}
        - Meta-Learner 綜合評估勝率：{blackboard.prob_final:.2%}

        【任務要求】
        1. 解釋為什麼系統會做出「{blackboard.action_decision}」的決策，若有損益請進行簡短點評。
        2. 比較左右腦的勝率差異，分析技術指標與K線型態是否有共識或分歧。
        3. 語氣請保持專業、客觀，避免過度承諾獲利。
        """

        # 呼叫 Gemini API (這裡先用 Mock 替代，等你接上真實 API)
        try:
            mock_response = f"【模擬 Gemini 回覆】\n根據綜合勝率 {blackboard.prob_final:.2%}，系統目前判定為 {blackboard.action_decision}。XGBoost 與 DL 模型的數據顯示..."
            blackboard.gemini_reasoning = mock_response
            dbg.log("Gemini 報告生成完畢！")
            return NodeState.SUCCESS

        except Exception as e:
            dbg.error(f"Gemini API 呼叫失敗: {e}")
            blackboard.gemini_reasoning = "API 呼叫失敗，無法生成報告。"
            return NodeState.FAILURE
