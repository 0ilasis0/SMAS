from bt.blackboard import Blackboard
from bt.const import ConditionCol, DecisionAction, LLMSentimentCol
from bt.core import BaseNode, NodeState
from bt.params import ConsiderConfig, LLMParams, TaxRate
from debug import dbg
from ml.const import FeatureCol


class CheckNotOverheatedNode(BaseNode):
    """
    防線 2：追高防呆鎖 (IDSS 自訂引擎版)。
    如果股票短線漲太多 (近 5 日)、乖離過大 (月線)，強制禁止買進。
    """
    def __init__(self, max_return_5d: float, max_bias_20: float, name=ConditionCol.CHECK_NOT_OVERHEATED):
        super().__init__(name)
        self.max_return_5d = max_return_5d
        self.max_bias_20 = max_bias_20

    def tick(self, blackboard: Blackboard) -> NodeState:
        # 1. 取得黑板上的防呆數據
        ret_5d = getattr(blackboard, FeatureCol.RETURN_5D, 0.0)
        bias_20 = getattr(blackboard, FeatureCol.BIAS_MONTH, 0.0)

        # 2. 判斷是否過熱
        if ret_5d > self.max_return_5d or bias_20 > self.max_bias_20:

            # 物理煞車：強制寫入 HOLD 並記錄警告
            blackboard.action_decision = DecisionAction.HOLD
            blackboard.gemini_reasoning = f"⚠️ 系統觸發追高防呆鎖 (近5日漲幅: {ret_5d:.1%}, 月線乖離: {bias_20:.1%})。為防主力出貨，強制阻斷買進訊號。"

            return NodeState.FAILURE

        return NodeState.SUCCESS


class CheckCooldownNode(BaseNode):
    def __init__(self, cooldown_days: int, name: str = ConditionCol.CHECK_COOLDOWN):
        super().__init__(name)
        self.cooldown_days = cooldown_days

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.cooldown_timer > 0:
            dbg.war(f"🧊 [進攻取消] 系統處於停損冷卻期 (剩餘 {blackboard.cooldown_timer} 天)，拒絕進場！(FAILURE)")
            return NodeState.FAILURE

        return NodeState.SUCCESS


class CheckTrendFilterNode(BaseNode):
    """
    大趨勢過濾節點。
    如果目前股價處於明確的空頭趨勢 (例如跌破月線/季線)，拒絕做多。
    (這裡我們用一個簡單的近似法：如果股價距離過去的高點跌幅超過 15%，視為空頭瀑布)
    """
    def __init__(self, safe_threshold: float, name: str = ConditionCol.CHECK_TREND_FILTER):
        super().__init__(name)
        self.safe_threshold = safe_threshold

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.prob_market_safe < self.safe_threshold:
            dbg.war(f"📉 [進攻取消] 大盤防禦雷達啟動 (安全度僅 {blackboard.prob_market_safe:.2%})，拒絕逆勢接刀！")
            return NodeState.FAILURE
        return NodeState.SUCCESS


class CheckSentimentFilterNode(BaseNode):
    """
    LLM 情緒防禦過濾節點 (News Sentiment 守門員)。
    若近期新聞被判定為重大利空 (分數低於門檻)，拒絕多單進場。
    """
    def __init__(self, min_score: int, name: str = ConditionCol.CHECK_SENTIMENT_FILTER):
        super().__init__(name)
        self.min_score = min_score

    def tick(self, blackboard: Blackboard) -> NodeState:
        current_score = getattr(blackboard, LLMSentimentCol.SCORE, LLMParams.DEFAULT_SENTIMENT_SCORE)
        current_reason = getattr(blackboard, LLMSentimentCol.REASON, '無相關新聞或未啟用 LLM')

        if current_score < self.min_score:
            dbg.war(f"📰 [進攻取消] LLM 判讀新聞為重大利空 (分數: {current_score}/10, 理由: {current_reason})，拒絕買進！")
            return NodeState.FAILURE

        return NodeState.SUCCESS


class CheckSellSentimentFilterNode(BaseNode):
    """
    LLM 停利防禦過濾節點 (News Sentiment 賣出守門員)。
    若近期新聞被判定為極度利多 (分數 >= 門檻)，則退回賣出請求，繼續讓獲利奔跑。
    ⚠️ 嚴格警告：此節點絕對不可放在「強制停損 (Stop-Loss)」的邏輯分支出去！
    """
    def __init__(self, block_score: int, name: str = ConditionCol.CHECK_SELL_SENTIMENT_FILTER):
        super().__init__(name)
        self.block_score = block_score

    def tick(self, blackboard: Blackboard) -> NodeState:
        current_score = getattr(blackboard, LLMSentimentCol.SCORE, 5)
        current_reason = getattr(blackboard, LLMSentimentCol.REASON, '無相關新聞或未啟用 LLM')

        # 如果情緒極度樂觀，阻止這次的賣出
        if current_score >= self.block_score:
            dbg.log(f"🛡️ [停利攔截] LLM 判讀新聞為極度利多 (分數: {current_score}/10, 理由: {current_reason})。阻擋 AI 賣出訊號，讓獲利繼續奔跑！")
            return NodeState.FAILURE

        # 情緒普普或看跌，同意放行技術面賣出
        return NodeState.SUCCESS


class CheckGapLimitNode(BaseNode):
    """
    檢查隔日開盤跳空幅度是否過大。
    用來防止 AI 判定買進，但隔天直接開漲停或大幅跳空，導致買在極高風險的位置。
    """
    def __init__(self, max_gap_ratio: float, name: str = ConditionCol.CHECK_GAP_LIMIT):
        super().__init__(name)
        # 預設：如果明天開盤跳空大於 ~%，就放棄買進
        self.max_gap_ratio = max_gap_ratio

    def tick(self, blackboard: Blackboard) -> NodeState:
        today_close = blackboard.current_price
        tomorrow_open = blackboard.executable_price

        if today_close <= 0:
            return NodeState.FAILURE

        gap_ratio = (tomorrow_open - today_close) / today_close

        if gap_ratio > self.max_gap_ratio:
            dbg.war(f"🛡️ [進攻取消] 明日開盤跳空達 {gap_ratio:.2%}，超過容忍值 {self.max_gap_ratio:.2%}，拒絕追高！(FAILURE)")
            return NodeState.FAILURE

        return NodeState.SUCCESS


class CheckNotPartialTakenNode(BaseNode):
    """
    檢查是否「尚未」執行過部分停利。
    用來防止「碎肉機陷阱」(避免每天都在賣一半)。
    """
    def __init__(self, name: str = ConditionCol.CHECK_NOT_PARTIAL_TAKEN):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        # 如果已經部分停利過了，就回傳 FAILURE 阻斷路徑
        if blackboard.is_partial_profit_taken:
            return NodeState.FAILURE
        return NodeState.SUCCESS


class CheckEntryCountLimitNode(BaseNode):
    """
    檢查加碼次數是否未達上限。
    用來防止「無底洞奈米加碼」。
    """
    def __init__(self, max_entries: int, name: str = ConditionCol.CHECK_ENTRY_COUNT_LIMIT):
        super().__init__(name)
        self.max_entries = max_entries

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.entry_count < self.max_entries:
            return NodeState.SUCCESS

        return NodeState.FAILURE


# 部位與狀態檢查
class CheckHasPositionNode(BaseNode):
    """
    檢查目前是否持有部位。
    若持有部位回傳 SUCCESS，若空手回傳 FAILURE。
    """
    def __init__(self, name: str = ConditionCol.CHECK_HAS_POSITION):
        super().__init__(name)

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position > 0:
            return NodeState.SUCCESS
        return NodeState.FAILURE


# 勝率訊號檢查
class CheckBuySignalNode(BaseNode):
    """
    檢查綜合勝率是否達到「買進門檻」。
    """
    def __init__(self, threshold: float = ConsiderConfig.BUY_THRESHOLD, name: str = ConditionCol.CHECK_BUY_SIGNAL):
        super().__init__(name)
        self.threshold = threshold

    def tick(self, blackboard: Blackboard) -> NodeState:
        prob = blackboard.prob_final
        if prob >= self.threshold:
            dbg.log(f"🔎 [條件檢查] 勝率 {prob:.2%} >= 買進門檻 ➔ 允許買進 (SUCCESS)")
            return NodeState.SUCCESS
        # dbg.log(f"🔎 [條件檢查] 勝率 {prob:.2%} <= 買進門檻 ➔ 不買進")
        return NodeState.FAILURE


class CheckSellSignalNode(BaseNode):
    """
    檢查綜合勝率是否跌破「賣出門檻」（例如模型極度看空）。
    """
    def __init__(self, threshold: float = ConsiderConfig.SELL_THRESHOLD, name: str = ConditionCol.CHECK_SELL_SIGNAL):
        super().__init__(name)
        self.threshold = threshold

    def tick(self, blackboard: Blackboard) -> NodeState:
        prob = blackboard.prob_final
        # 如果勝率小於等於賣出門檻，代表模型看跌，產生賣出訊號
        if prob <= self.threshold:
            dbg.log(f"🔎 [條件檢查] 勝率 {prob:.2%} <= 賣出門檻 {self.threshold:.2%} ➔ 允許賣出 (SUCCESS)")
            return NodeState.SUCCESS

        return NodeState.FAILURE


# 風險控管 (停損停利) 檢查
class CheckStopLossNode(BaseNode):
    """
    檢查是否觸發停損。
    若虧損比例超過容忍值，回傳 SUCCESS (代表條件成立，觸發後續賣出動作)。
    """
    def __init__(self, loss_tolerance: float, cooldown_days: int, name: str = ConditionCol.CHECK_STOP_LOSS):
        super().__init__(name)
        self.loss_tolerance = loss_tolerance
        self.cooldown_days = cooldown_days

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position <= 0: return NodeState.FAILURE

        close_return = blackboard.estimated_return_rate
        open_revenue = blackboard.position * blackboard.executable_price
        total_cost = blackboard.position * blackboard.avg_cost

        fee = max(TaxRate.MIN_FEE, open_revenue * TaxRate.FEE_RATE)
        tax = open_revenue * TaxRate.TAX_RATE
        open_return = (open_revenue - fee - tax - total_cost) / total_cost if total_cost > 0 else 0

        if close_return <= self.loss_tolerance or open_return <= self.loss_tolerance:
            trigger_price = blackboard.current_price if close_return <= self.loss_tolerance else blackboard.executable_price
            dbg.war(f"⚠️ [風險控管] 觸發強制停損！觸價: {trigger_price:.2f} 預估報酬率: {min(close_return, open_return):.2%} <= 容忍底線 {self.loss_tolerance:.2%} (SUCCESS)")

            blackboard.cooldown_timer = self.cooldown_days
            return NodeState.SUCCESS

        return NodeState.FAILURE


class CheckTakeProfitNode(BaseNode):
    """
    檢查是否觸發停利。
    若獲利比例超過目標值，回傳 SUCCESS (代表條件成立，觸發後續賣出動作)。
    """
    def __init__(self, profit_target: float, name: str = ConditionCol.CHECK_TAKE_PROFIT):
        super().__init__(name)
        self.profit_target = profit_target

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position <= 0: return NodeState.FAILURE

        real_return = blackboard.estimated_return_rate

        if real_return >= self.profit_target:
            dbg.log(f"🎉 [風險控管] 觸發停利！真實報酬率 {real_return:.2%} >= 目標利潤 {self.profit_target:.2%} (SUCCESS)")
            return NodeState.SUCCESS
        return NodeState.FAILURE


class CheckTrailingStopNode(BaseNode):
    """
    移動停損 (Trailing Stop) 檢查。
    從持倉以來的最高點回落超過設定比例，即觸發出場 (鎖住利潤或限制虧損)。
    """
    def __init__(self, drawdown_tolerance: float, cooldown_days: int, name: str = ConditionCol.CHECK_TRAILING_STOP):
        super().__init__(name)
        self.drawdown_tolerance = drawdown_tolerance
        self.cooldown_days = cooldown_days

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position <= 0 or blackboard.highest_price <= 0:
            return NodeState.FAILURE

        highest_price = blackboard.highest_price
        close_drawdown = (blackboard.current_price - highest_price) / highest_price
        open_drawdown = (blackboard.executable_price - highest_price) / highest_price

        if close_drawdown <= self.drawdown_tolerance or open_drawdown <= self.drawdown_tolerance:
            actual_drawdown = min(close_drawdown, open_drawdown)
            dbg.war(f"🛡️ [風險控管] 觸發移動停損！最高點 {highest_price:.2f} 回落 {actual_drawdown:.2%} <= 容忍底線 {self.drawdown_tolerance:.2%} (SUCCESS)")

            blackboard.cooldown_timer = self.cooldown_days
            return NodeState.SUCCESS

        return NodeState.FAILURE
