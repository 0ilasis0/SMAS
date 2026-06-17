from dataclasses import dataclass


# 用於買賣限制倍率(單位：股)
@dataclass(frozen=True)
class OrderRule:
    """存放下單與交易相關的規則與限制"""
    LOT_SIZE: int = 100  # 買賣限制倍率(單位：股)