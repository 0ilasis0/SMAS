import streamlit as st
from bt.strategy_config import TradingPersona
from ml.model.llm_oracle import TradingMode
from ui.state import on_ticker_change, reset_result

def render_sidebar() -> tuple[TradingPersona, TradingMode]:
    """渲染左側邊欄，回傳使用者選擇的戰術與模式"""
    with st.sidebar:
        st.title("⚙️ IDSS 控制台")
        st.markdown("---")

        st.selectbox(
            "📌 自選股專案區",
            st.session_state.watch_list,
            index=st.session_state.watch_list.index(st.session_state.current_ticker),
            key="new_ticker",
            on_change=on_ticker_change
        )
        st.markdown("---")

        persona_mapping = {
            "激進型 (AGGRESSIVE)": TradingPersona.AGGRESSIVE,
            "穩健型 (MODERATE)": TradingPersona.MODERATE,
            "保守型 (CONSERVATIVE)": TradingPersona.CONSERVATIVE
        }
        selected_persona_str = st.selectbox(
            "🧠 戰術性格", list(persona_mapping.keys()), index=1, on_change=reset_result
        )

        mode_mapping = {
            "波段模式 (SWING) - 啟用 AI 新聞情緒": TradingMode.SWING,
            "當沖模式 (DAY_TRADE) - 極速技術面決策": TradingMode.DAY_TRADE
        }
        selected_mode_str = st.selectbox(
            "⚡ 交易模式", list(mode_mapping.keys()), index=0, on_change=reset_result
        )

        st.markdown("---")
        st.caption("ℹ️ 資料同步與資金管理功能將於 V2 開放")

        if st.button("🔄 重新載入最新資料", use_container_width=True):
            st.session_state.ctrl_live = None
            st.session_state.ctrl_bt = None
            reset_result()
            st.rerun()

        return persona_mapping[selected_persona_str], mode_mapping[selected_mode_str]