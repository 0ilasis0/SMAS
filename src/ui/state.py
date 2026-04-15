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

def load_watchlist() -> list:
    """從本地讀取自選股記憶，若無則回傳預設值"""
    if os.path.exists(PathConfig.WATCHLIST):
        try:
            with open(PathConfig.WATCHLIST, 'r', encoding=EncodingConst.UTF8.value) as f:
                return json.load(f)
        except Exception:
            pass
    return ["2330.TW"]

# def save_watchlist(watchlist: list):
#     """將自選股狀態寫入本地硬碟保存"""
#     # 確保資料夾存在
#     os.makedirs(os.path.dirname(PathConfig.WATCHLIST), exist_ok=True)
#     with open(PathConfig.WATCHLIST, 'w', encoding=EncodingConst.UTF8.value) as f:
#         json.dump(watchlist, f, ensure_ascii=False, indent=4)

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
    if SessionKey.WATCH_LIST.value not in st.session_state:
        st.session_state[SessionKey.WATCH_LIST.value] = load_watchlist()

    if SessionKey.CURRENT_TICKER.value not in st.session_state:
        watch_list = st.session_state.get(SessionKey.WATCH_LIST.value, [])
        st.session_state[SessionKey.CURRENT_TICKER.value] = watch_list[0] if watch_list else None

    if SessionKey.USER_SETTINGS.value not in st.session_state:
        st.session_state[SessionKey.USER_SETTINGS.value] = load_settings()

    if SessionKey.PORTFOLIO.value not in st.session_state:
        st.session_state[SessionKey.PORTFOLIO.value] = load_portfolio()

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
