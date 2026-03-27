from bt.blackboard import Blackboard
from bt.core import BaseNode, NodeState
from bt.variable import BtVar, DecisionAction, ExecuteCol
from debug import dbg


class TaxRate:
    ''' 台灣股市基礎費率設定 (可依券商折讓自行調整) '''
    FEE_RATE: float = 0.001425  # 券商手續費率 (買賣皆收)
    TAX_RATE: float = 0.003     # 證券交易稅率 (僅賣出收取)
    MIN_FEE: float = 20.0       # 手續費低消


# 交易動作節點 (虛擬交易執行)
class ExecuteBuyNode(BaseNode):
    """執行買進動作，並更新黑板上的資金與部位"""
    def __init__(self, name: str = ExecuteCol.BUY):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        price = blackboard.current_price
        cash = blackboard.cash

        if price <= 0:
            dbg.error("股價異常，無法執行買進！")
            return NodeState.FAILURE

        max_shares_prop = int(cash // (price * (1 + TaxRate.FEE_RATE)))
        max_shares_min = int((cash - 20) // price)
        shares_to_buy = min(max_shares_prop, max_shares_min)

        if shares_to_buy <= 0:
            dbg.war("資金不足以購買任何零股/整股！")
            return NodeState.FAILURE

        raw_cost = shares_to_buy * price
        fee = max(TaxRate.MIN_FEE, raw_cost * TaxRate.FEE_RATE)
        total_cost = raw_cost + fee

        if total_cost > cash:
            dbg.war("加計手續費後資金不足！")
            return NodeState.FAILURE

        old_total_cost = blackboard.position * blackboard.avg_cost
        blackboard.cash -= total_cost

        new_position = blackboard.position + shares_to_buy
        blackboard.avg_cost = (old_total_cost + total_cost) / new_position
        blackboard.position = new_position

        blackboard.action_decision = DecisionAction.BUY

        # 寫入黑板，供 Gemini 報告使用
        blackboard.last_trade_shares = shares_to_buy
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = 0.0

        dbg.log(f"🟢 [交易執行] 買進 {shares_to_buy} 股，成交價 {price:.2f}。手續費 {fee:.0f}。剩餘資金: {blackboard.cash:.2f}")
        return NodeState.SUCCESS


class ExecuteSellNode(BaseNode):
    """執行賣出動作，清空部位並換回現金"""
    def __init__(self, name: str = ExecuteCol.SELL):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        price = blackboard.current_price
        position = blackboard.position

        if position <= 0:
            dbg.error("目前無部位，無法執行賣出！")
            return NodeState.FAILURE

        # 計算賣出實拿金額與損益
        raw_revenue = position * price
        fee = max(TaxRate.MIN_FEE, raw_revenue * TaxRate.FEE_RATE)
        tax = raw_revenue * TaxRate.TAX_RATE
        actual_revenue = raw_revenue - fee - tax

        profit = actual_revenue - (position * blackboard.avg_cost)

        blackboard.cash += actual_revenue
        blackboard.position = 0
        blackboard.avg_cost = 0.0
        blackboard.action_decision = DecisionAction.SELL

        # 寫入黑板，供 Gemini 報告使用
        blackboard.last_trade_shares = position
        blackboard.last_trade_price = price
        blackboard.last_trade_profit = profit

        dbg.log(f"🔴 [交易執行] 賣出 {position} 股，成交價 {price:.2f}。淨損益: {profit:.2f}。目前資金: {blackboard.cash:.2f}")
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
