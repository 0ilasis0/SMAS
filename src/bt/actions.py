from bt.blackboard import Blackboard
from bt.const import BtVar, DecisionAction, ExecuteCol
from bt.core import BaseNode, NodeState
from bt.params import ConsiderVar, TaxRate
from debug import dbg


# 交易動作節點 (虛擬交易執行)
class ExecuteBuyNode(BaseNode):
    """
    執行買進動作，並更新黑板上的資金與部位。
    :param capital_ratio: 動用可用資金的比例 (0.0 ~ 1.0)，1.0 為 All-in
    """
    def __init__(self, name: str = ExecuteCol.BUY, capital_ratio: float = ConsiderVar.CAPITAL_RATIO):
        super().__init__(name)
        self.capital_ratio = capital_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        price = blackboard.current_price
        usable_cash = blackboard.cash * self.capital_ratio

        if price <= 0:
            dbg.error("股價異常，無法執行買進！")
            return NodeState.FAILURE

        max_shares_prop = int(usable_cash // (price * (1 + TaxRate.FEE_RATE)))
        max_shares_min = int((usable_cash - TaxRate.MIN_FEE) // price)
        shares_to_buy = max(0, min(max_shares_prop, max_shares_min))

        if shares_to_buy <= 20:
            dbg.war("資金不足以購買任何零股/整股！")
            return NodeState.FAILURE

        # 交易成本計算
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

        if old_total_cost == 0:
            blackboard.highest_price = price

        dbg.log(f"🟢 [交易執行] 動用 {self.capital_ratio:.0%} 資金買進 {shares_to_buy} 股，成交價 {price:.2f}。剩餘總資金: {blackboard.cash:.2f}")
        return NodeState.SUCCESS


class ExecuteSellNode(BaseNode):
    """
    執行賣出動作，換回現金。
    :param position_ratio: 賣出目前部位的比例 (0.0 ~ 1.0)，預設 1.0 為全數出清
    """
    def __init__(self, name: str = ExecuteCol.SELL, position_ratio: float = ConsiderVar.POSITION_RATIO):
        super().__init__(name)
        self.position_ratio = position_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        price = blackboard.current_price
        position = blackboard.position

        shares_to_sell = int(position * self.position_ratio)

        if shares_to_sell <= 0:
            dbg.war("賣出比例換算股數不足 1 股，無法賣出！")
            return NodeState.FAILURE

        # 計算賣出實拿金額與損益
        raw_revenue = shares_to_sell * price
        fee = max(TaxRate.MIN_FEE, raw_revenue * TaxRate.FEE_RATE)
        tax = raw_revenue * TaxRate.TAX_RATE
        actual_revenue = raw_revenue - fee - tax

        # 損益 = 賣出實拿 - (賣出股數 * 平均成本)
        profit = actual_revenue - (shares_to_sell * blackboard.avg_cost)

        # 更新黑板帳戶狀態
        blackboard.cash += actual_revenue
        blackboard.position -= shares_to_sell

        if blackboard.position == 0:
            blackboard.avg_cost = 0.0
            blackboard.highest_price = 0.0

        blackboard.action_decision = DecisionAction.SELL

        # 寫入黑板供 Gemini 使用
        blackboard.last_trade_shares = shares_to_sell
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = profit

        dbg.log(f"🔴 [交易執行] 賣出 {self.position_ratio:.0%} 部位 ({shares_to_sell} 股)，成交價 {price:.2f}。淨損益: {profit:.2f}。目前資金: {blackboard.cash:.2f}")
        return NodeState.SUCCESS


class ExecuteHoldNode(BaseNode):
    """保持觀望，不進行任何交易"""
    def __init__(self, name: str = ExecuteCol.HOLD):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        blackboard.action_decision = DecisionAction.HOLD
        dbg.log("⚪ [交易執行] 維持現狀 (HOLD)。")
        return NodeState.SUCCESS


# AI 報告生成節點 (呼叫 Gemini)
class GenerateGeminiReportNode(BaseNode):
    """
    負責收集黑板上的所有資訊，打包成 Prompt，並呼叫 Gemini 產出分析報告。
    """
    def __init__(self, name: str = BtVar.GENERATE_GEMINI_REPORT):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        dbg.log("🧠 正在呼叫 Gemini 生成決策分析報告...")

        trade_info_str = ""
        if blackboard.action_decision == DecisionAction.BUY:
            trade_info_str = f"- 交易執行：系統買進了 {blackboard.last_trade_shares} 股，成交價 {blackboard.last_trade_price:.2f}。"
        elif blackboard.action_decision == DecisionAction.SELL:
            trade_info_str = f"- 交易執行：系統全數出清 {blackboard.last_trade_shares} 股，成交價 {blackboard.last_trade_price:.2f}，本次交易淨損益為 {blackboard.last_trade_profit:.2f} 元。"
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
            dbg.log("✅ Gemini 報告生成完畢！")
            return NodeState.SUCCESS

        except Exception as e:
            dbg.error(f"Gemini API 呼叫失敗: {e}")
            blackboard.gemini_reasoning = "API 呼叫失敗，無法生成報告。"
            return NodeState.FAILURE
