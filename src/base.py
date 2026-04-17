import traceback

import pandas as pd
from dotenv import dotenv_values
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
        env_path = PathConfig.GEMINI_KEY
        if not env_path.exists():
            dbg.error(f"❌ 找不到設定檔: {env_path}")
            return []

        try:
            # 直接從檔案解析成字典，徹底擺脫 os.environ 的快取干擾
            env_dict = dotenv_values(dotenv_path=env_path)

            # 雙重保險防呆：同時找尋「複數」與「單數」的變數名稱
            keys_str = env_dict.get(GlobalVar.GEMINI_API_KEYS)
            if not keys_str:
                keys_str = env_dict.get("GEMINI_API_KEY", "")

            if keys_str is None:
                keys_str = ""

            # 清理字串
            clean_str = str(keys_str).replace('"', '').replace("'", "").replace(" ", "").replace("　", "")
            keys_list = [k for k in clean_str.split(",") if k]

            # 終極防護網：如果真的還是找不到，印出檔案內容
            if not keys_list:
                with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read(150) # 只讀前 150 個字，避免印出太多

                dbg.error(f"❌ 檔案 {env_path} 明明存在，但就是抓不到 Key！")
                dbg.error(f"👉 請檢查您的變數名稱，必須是 GEMINI_API_KEYS=您的金鑰")
                dbg.error(f"👉 以下是系統實際讀取到的【檔案前 150 個字元】：\n{repr(raw_content)}")
                return []

            return keys_list

        except Exception as e:
            error_details = traceback.format_exc()
            dbg.error(f"❌ 讀取 {env_path} 發生深層崩潰: {e}\n{error_details}")
            return []



#
# 狀態總管理
#
class CentralManager:
    def __init__(self):
        self.running: bool = False

central_mg = CentralManager()
