from dataclasses import dataclass


class GlobalVar:
    GEMINI_API_KEYS: str = "GEMINI_API_KEYS"


@dataclass(frozen=True)
class IDSSTrain:
    MAX_TIME = 240
    MIX_TIME = 60
    STEP_TIME = 10
