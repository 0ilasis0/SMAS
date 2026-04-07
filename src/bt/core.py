from abc import ABC, abstractmethod
from enum import Enum, auto

from bt.blackboard import Blackboard


class NodeState(Enum):
    """行為樹節點的執行狀態"""
    SUCCESS = auto()  # 執行成功 / 條件符合
    FAILURE = auto()  # 執行失敗 / 條件不符
    RUNNING = auto()  # 執行中 (在回合制定時 K 線交易中較少用到，保留以供未來擴充)


# ==========================================
# 1. 抽象基底節點
# ==========================================
class BaseNode(ABC):
    """
    所有行為樹節點的抽象基底類別。
    """
    def __init__(self, name: str = 'base_node'):
        self.name = name

    @abstractmethod
    def tick(self, blackboard: Blackboard) -> NodeState:
        """
        每次系統心跳 (Tick) 時執行的邏輯。
        子類別必須實作此方法，並回傳 SUCCESS, FAILURE 或 RUNNING。
        """
        pass


# ==========================================
# 2. 語意化葉節點 (Leaf Nodes)
# ==========================================
class ConditionNode(BaseNode):
    """
    條件節點 (葉節點)。
    語意上僅負責「讀取與檢查」Blackboard 的狀態，不應對系統狀態進行修改。
    """
    def __init__(self, name: str = 'condition_node'):
        super().__init__(name)


class ActionNode(BaseNode):
    """
    動作節點 (葉節點)。
    語意上負責「執行實體動作」或「修改」Blackboard 的狀態 (例如產生交易訊號)。
    """
    def __init__(self, name: str = 'action_node'):
        super().__init__(name)


# ==========================================
# 3. 組合節點 (Composite Nodes)
# ==========================================
class Sequence(BaseNode):
    """
    序列節點 (AND 邏輯)。
    由左至右執行子節點，只要有一個子節點回傳 FAILURE，就立刻中斷並回傳 FAILURE。
    必須所有子節點都回傳 SUCCESS，此節點才算 SUCCESS。
    (用途範例：檢查勝率達標 -> 檢查資金足夠 -> 執行買進)
    """
    def __init__(self, name: str, children: list[BaseNode]):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard: Blackboard) -> NodeState:
        for child in self.children:
            state = child.tick(blackboard)

            if state == NodeState.FAILURE:
                return NodeState.FAILURE
            elif state == NodeState.RUNNING:
                return NodeState.RUNNING

        return NodeState.SUCCESS


class Selector(BaseNode):
    """
    選擇節點 (OR 邏輯)。
    由左至右執行子節點，只要有一個子節點回傳 SUCCESS，就立刻中斷並回傳 SUCCESS。
    必須所有子節點都回傳 FAILURE，此節點才算 FAILURE。
    (用途範例：遇到危機時，嘗試停損賣出 -> 嘗試減碼 -> 都不行只好發送警告)
    """
    def __init__(self, name: str, children: list[BaseNode]):
        super().__init__(name)
        self.children = children

    def tick(self, blackboard: Blackboard) -> NodeState:
        for child in self.children:
            state = child.tick(blackboard)
            if state != NodeState.FAILURE:
                return state

        return NodeState.FAILURE


# ==========================================
# 4. 裝飾節點 (Decorator Nodes)
# ==========================================
class Inverter(BaseNode):
    """
    反轉節點 (NOT 邏輯)。
    只能包含一個子節點。將子節點的 SUCCESS 轉為 FAILURE，FAILURE 轉為 SUCCESS。
    (用途範例：Inverter(檢查是否持有部位) -> 變成「確認目前為空手」)
    """
    def __init__(self, name: str, child: BaseNode):
        if not isinstance(child, BaseNode):
            raise TypeError(f"{child} must be a BaseNode")
        super().__init__(name)
        self.child = child

    def tick(self, blackboard: Blackboard) -> NodeState:
        state = self.child.tick(blackboard)
        if state == NodeState.SUCCESS:
            return NodeState.FAILURE
        elif state == NodeState.FAILURE:
            return NodeState.SUCCESS
        return state


class ForceSuccess(BaseNode):
    """
    強制成功節點 (Always Succeed)。
    無論子節點回傳什麼，此節點永遠回傳 SUCCESS。
    (用途範例：用在 Sequence 內部包裝「非必要」的動作，如寫入次要 Log。即使寫入失敗，也不會阻斷主流程的執行)
    """
    def __init__(self, name: str, child: BaseNode):
        if not isinstance(child, BaseNode):
            raise TypeError(f"{child} must be a BaseNode")
        super().__init__(name)
        self.child = child

    def tick(self, blackboard: Blackboard) -> NodeState:
        self.child.tick(blackboard)
        # 無論 child 是 SUCCESS, FAILURE 或 RUNNING，直接蓋掉回傳 SUCCESS
        return NodeState.SUCCESS
