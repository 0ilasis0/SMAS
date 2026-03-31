from dataclasses import dataclass


class GlobalVar:
    GEMINI_API_KEYS: str = "GEMINI_API_KEYS"


@dataclass(frozen=True)
class IDSSParams:
    TEST_MAX_TIME = 240
    TEST_MIX_TIME = 60
    TEST_STEP_TIME = 10
