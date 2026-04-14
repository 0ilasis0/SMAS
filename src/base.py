import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray

from const import GlobalVar
from debug import dbg
from path import PathConfig


class FrozenMeta(type):
    def __setattr__(cls, name, value):
        # 攔截對類別屬性的賦值行為
        raise AttributeError(f"無法修改常數屬性: '{name}'")


class MathTool:
    @staticmethod
    def clamp(val: int, min_val: int, max_val: int):
        if max_val < min_val:
            dbg.war("clamp 參數顛倒，已自動交換 min/max")
            min_val, max_val = max_val, min_val
        return max(min(val, max_val), min_val)

class MLTool:
    @staticmethod
    def calculate_scale_weight(y: pd.Series | NDArray) -> float:
        """計算正負樣本不平衡的權重比例"""
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        return float(neg_count / pos_count) if pos_count > 0 else 1.0


class KeyManager:
    """自動從 .env 讀取並管理 API Key 池"""
    @staticmethod
    def get_gemini_keys() -> list[str]:
        # 指定從你定義的 PathConfig 載入 .env
        env_path = PathConfig.GEMINI_KEY
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            dbg.error(f"❌ 找不到設定檔: {env_path}")
            return []

        # 讀取長字串
        keys_str = os.getenv(GlobalVar.GEMINI_API_KEYS, "")

        # 把可能殘留的單引號、雙引號、甚至是全形空白全部拔除
        clean_str = keys_str.replace('"', '').replace("'", "").replace(" ", "").replace("　", "")

        # 根據逗號切割成 List
        keys_list = [k for k in clean_str.split(",") if k]

        if not keys_list:
            dbg.error(f"❌ 在 {env_path} 中找不到任何有效 API Key！請檢查變數名稱是否為 {GlobalVar.GEMINI_API_KEYS} 或 GEMINI_API_KEY")

        return keys_list



#
# 狀態總管理
#
class CentralManager:
    def __init__(self):
        self.running: bool = False

central_mg = CentralManager()
