import json
import os

import streamlit as st

from path import PathConfig
from ui.const import EncodingConst, Page
from ui.portfolio import load_portfolio


def load_settings() -> dict:
    if os.path.exists(PathConfig.SETTINGS):
        try:
            with open(PathConfig.SETTINGS, 'r', encoding=EncodingConst.UTF8) as f:
                return json.load(f)
        except Exception:
            pass
    return {"persona": "穩健型 (MODERATE)", "mode": "波段模式 (SWING)"}

def save_settings(persona: str, mode: str):
    os.makedirs(os.path.dirname(PathConfig.SETTINGS), exist_ok=True)
    with open(PathConfig.SETTINGS,  'w', encoding=EncodingConst.UTF8) as f:
        json.dump({"persona": persona, "mode": mode}, f, ensure_ascii=False, indent=4)

def load_watchlist() -> list:
    """從本地讀取自選股記憶，若無則回傳預設值"""
    if os.path.exists(PathConfig.WATCHLIST):
        try:
            with open(PathConfig.WATCHLIST, 'r', encoding=EncodingConst.UTF8) as f:
                return json.load(f)
        except Exception:
            pass
    return ["2330.TW"]

def save_watchlist(watchlist: list):
    """將自選股狀態寫入本地硬碟保存"""
    # 確保資料夾存在
    os.makedirs(os.path.dirname(PathConfig.WATCHLIST), exist_ok=True)
    with open(PathConfig.WATCHLIST, 'w', encoding=EncodingConst.UTF8) as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=4)

def reset_result():
    """清除前一次的預測結果"""
    st.session_state.last_result = None

def on_ticker_change(new_ticker: str):
    """當使用者切換股票時，清空雙大腦與畫面狀態"""
    st.session_state.current_ticker = new_ticker
    st.session_state.ctrl_live = None
    st.session_state.ctrl_bt = None
    reset_result()

def init_session_state():
    """初始化系統的全域狀態 (支援本地記憶)"""
    if 'watch_list' not in st.session_state:
        st.session_state.watch_list = load_watchlist()

    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = st.session_state.watch_list[0] if st.session_state.watch_list else None

    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = load_settings()
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = Page.DASHBOARD

    if 'ctrl_live' not in st.session_state:
        st.session_state.ctrl_live = None
    if 'ctrl_bt' not in st.session_state:
        st.session_state.ctrl_bt = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    if 'is_training' not in st.session_state:
        st.session_state.is_training = False