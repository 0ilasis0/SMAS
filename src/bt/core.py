from abc import ABC, abstractmethod
from enum import Enum, auto

from bt.blackboard import Blackboard
from bt.const import BtVar


class NodeState(Enum):
    """行為樹節點的執行狀態"""
    SUCCESS = auto()  # 執行成功
    FAILURE = auto()  # 執行失敗 / 條件不符
    RUNNING = auto()  # 執行中 (在回合制交易中較少用到，多用於連續動作)


class BaseNode(ABC):
    """
    所有行為樹節點的抽象基底類別。
    """
    def __init__(self, name: str = BtVar.BASE_NODE):
        self.name = name

    @abstractmethod
    def tick(self, blackboard: Blackboard) -> NodeState:
        """
        每次系統心跳 (Tick) 時執行的邏輯。
        子類別必須實作此方法，並回傳 SUCCESS, FAILURE 或 RUNNING。
        """
        pass


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
            if state != NodeState.SUCCESS:
                # 遇到 FAILURE 或 RUNNING，直接中斷並向上回傳
                return state
        # 全部子節點都成功
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
                # 遇到 SUCCESS 或 RUNNING，代表找到可行的路，直接向上回傳
                return state
        # 所有的備案都失敗了
        return NodeState.FAILURE
