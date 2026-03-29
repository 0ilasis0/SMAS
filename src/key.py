import os

from dotenv import load_dotenv

from debug import dbg
from path import PathConfig


class KeyManager:
    """自動從 .env 讀取並管理 API Key 池"""
    @staticmethod
    def get_gemini_keys() -> list[str]:
        # 指定從你定義的 PathConfig 載入 .env
        env_path = PathConfig.GEMINI_KEY
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        # 讀取長字串並分割成清單
        keys_str = os.getenv("GEMINI_API_KEYS", "")

        # 過濾掉空白，並轉成 List
        keys_list = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not keys_list:
            dbg.error(f"❌ 在 {env_path} 中找不到任何有效 API Key！")

        return keys_list
