import os

from dotenv import load_dotenv

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


class KeyManager:
    """自動從 .env 讀取並管理 API Key 池"""
    @staticmethod
    def get_gemini_keys() -> list[str]:
        # 指定從你定義的 PathConfig 載入 .env
        env_path = PathConfig.GEMINI_KEY
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        # 讀取長字串並分割成清單
        keys_str = os.getenv(GlobalVar.GEMINI_API_KEYS, "")

        # 過濾掉空白，並轉成 List
        keys_list = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not keys_list:
            dbg.error(f"❌ 在 {env_path} 中找不到任何有效 API Key！")

        return keys_list



#
# 狀態總管理
#
class CentralManager:
    def __init__(self):
        self.running: bool = False

central_mg = CentralManager()
