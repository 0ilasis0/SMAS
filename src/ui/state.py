import json
import os

import streamlit as st

from path import PathConfig
from ui.const import EncodingConst, Page, SessionKey
from ui.portfolio import load_portfolio


def load_settings() -> dict:
    if os.path.exists(PathConfig.SETTINGS):
        try:
            with open(PathConfig.SETTINGS, 'r', encoding=EncodingConst.UTF8.value) as f:
                return json.load(f)
        except Exception:
            pass
    return {"persona": "穩健型 (MODERATE)", "mode": "波段模式 (SWING)"}

def save_settings(persona: str, mode: str):
    os.makedirs(os.path.dirname(PathConfig.SETTINGS), exist_ok=True)
    with open(PathConfig.SETTINGS,  'w', encoding=EncodingConst.UTF8.value) as f:
        json.dump({"persona": persona, "mode": mode}, f, ensure_ascii=False, indent=4)

def reset_result():
    """清除前一次的預測結果"""
    st.session_state[SessionKey.LAST_RESULT.value] = None

def on_ticker_change(new_ticker: str):
    """當使用者切換股票時，清空雙大腦與畫面狀態"""
    st.session_state[SessionKey.CURRENT_TICKER.value] = new_ticker
    st.session_state[SessionKey.CTRL_LIVE.value] = None
    st.session_state[SessionKey.CTRL_BT.value] = None
    reset_result()

def init_session_state():
    """初始化系統的全域狀態 (支援本地記憶)"""

    # 優先載入設定與帳戶資產 (因為後續判斷會用到)
    if SessionKey.USER_SETTINGS.value not in st.session_state:
        st.session_state[SessionKey.USER_SETTINGS.value] = load_settings()

    if SessionKey.PORTFOLIO.value not in st.session_state:
        st.session_state[SessionKey.PORTFOLIO.value] = load_portfolio()

    # 預設為 None，讓 app.py 裡的「焦點防呆邏輯」自動去抓取組合包內的第一檔股票
    if SessionKey.CURRENT_TICKER.value not in st.session_state:
        st.session_state[SessionKey.CURRENT_TICKER.value] = None

    # 基礎 UI 狀態
    if SessionKey.CURRENT_PAGE.value not in st.session_state:
        st.session_state[SessionKey.CURRENT_PAGE.value] = Page.DASHBOARD.value

    if SessionKey.CTRL_LIVE.value not in st.session_state:
        st.session_state[SessionKey.CTRL_LIVE.value] = None

    if SessionKey.CTRL_BT.value not in st.session_state:
        st.session_state[SessionKey.CTRL_BT.value] = None

    if SessionKey.LAST_RESULT.value not in st.session_state:
        st.session_state[SessionKey.LAST_RESULT.value] = None

    if SessionKey.IS_TRAINING.value not in st.session_state:
        st.session_state[SessionKey.IS_TRAINING.value] = False

    if SessionKey.IS_GLOBAL_TRAINING.value not in st.session_state:
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = False