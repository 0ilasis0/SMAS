import streamlit as st


def reset_result():
    """清除前一次的預測結果"""
    st.session_state.last_result = None

def on_ticker_change():
    """當使用者切換股票時，清空雙大腦與畫面狀態"""
    st.session_state.current_ticker = st.session_state.new_ticker
    st.session_state.ctrl_live = None
    st.session_state.ctrl_bt = None
    reset_result()

def init_session_state():
    """初始化系統的所有必要全域狀態"""
    if 'watch_list' not in st.session_state:
        st.session_state.watch_list = ["3481.TW", "2388.TW", "5469.TW", "2337.TW"]
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = st.session_state.watch_list[0]

    # 雙軌控制器
    if 'ctrl_live' not in st.session_state:
        st.session_state.ctrl_live = None
    if 'ctrl_bt' not in st.session_state:
        st.session_state.ctrl_bt = None

    if 'last_result' not in st.session_state:
        st.session_state.last_result = None