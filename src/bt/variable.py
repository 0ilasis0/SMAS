from dataclasses import dataclass
from enum import StrEnum


class BtVar(StrEnum):
    BASE_NODE = "base_node"
    GENERATE_GEMINI_REPORT = "generate_gemini_report"



class ExecuteCol(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DecisionAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


