from bt.blackboard import Blackboard
from bt.const import ConditionCol
from bt.core import BaseNode, NodeState
from bt.params import ConsiderConfig
from debug import dbg


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
    def __init__(self, loss_tolerance: float, name: str = ConditionCol.CHECK_STOP_LOSS):
        super().__init__(name)
        # loss_tolerance 例如 -0.05 代表跌 5% 就停損
        self.loss_tolerance = loss_tolerance

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position <= 0: return NodeState.FAILURE

        real_return = blackboard.estimated_return_rate

        if real_return <= self.loss_tolerance:
            dbg.war(f"⚠️ [風險控管] 觸發停損！真實報酬率 {real_return:.2%} <= 容忍底線 {self.loss_tolerance:.2%} (SUCCESS)")
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
    def __init__(self, drawdown_tolerance: float, name: str = ConditionCol.CHECK_TRAILING_STOP):
        super().__init__(name)
        # drawdown_tolerance = -0.08 代表從最高點回落 8% 就強制出場
        self.drawdown_tolerance = drawdown_tolerance

    def tick(self, blackboard: Blackboard) -> NodeState:
        if blackboard.position <= 0 or blackboard.highest_price <= 0:
            return NodeState.FAILURE

        current_price = blackboard.current_price
        highest_price = blackboard.highest_price

        drawdown = (current_price - highest_price) / highest_price

        if drawdown <= self.drawdown_tolerance:
            dbg.war(f"🛡️ [風險控管] 觸發移動停損！從最高點 {highest_price:.2f} 回落 {drawdown:.2%} <= 容忍底線 {self.drawdown_tolerance:.2%} (SUCCESS)")
            return NodeState.SUCCESS

        return NodeState.FAILURE
