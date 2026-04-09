from dataclasses import dataclass
from enum import StrEnum


class GlobalVar:
    GEMINI_API_KEYS: str = "GEMINI_API_KEYS"


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    ORANGE = "orange"
    PURPLE = "purple"
    BLUE = "blue"
    WHITE = "white"
    GRAY = "gray"

@dataclass(frozen=True)
class GlobalParams:
    DEFAULT_ERROR = -9999999
