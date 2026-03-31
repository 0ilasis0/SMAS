from dataclasses import dataclass


@dataclass(frozen=True)
class IDSSParams:
    TRAIN_MAX_TIME = 240
    TRAIN_MIX_TIME = 60
    TRAIN_STEP_TIME = 10
