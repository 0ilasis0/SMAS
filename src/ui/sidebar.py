import streamlit as st

from bt.strategy_config import TradingPersona
from ml.model.llm_oracle import TradingMode
from ui.const import Page
from ui.state import (on_ticker_change, reset_result, save_settings,
                      save_watchlist)


def render_sidebar() -> tuple[TradingPersona, TradingMode]:
    # 讀取當前的鎖定狀態
    is_locked = st.session_state.get('is_training', False)

    with st.sidebar:
        st.title("⚙️ IDSS 控制台")

        if is_locked:
            st.warning("⏳ 系統正在進行 AI 模型訓練，控制台暫時鎖定。")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈 決策大廳", use_container_width=True, disabled=is_locked,
                         type="primary" if st.session_state.current_page == Page.DASHBOARD else "secondary"):
                st.session_state.current_page = Page.DASHBOARD
                st.rerun()
        with c2:
            if st.button("💼 資產管理", use_container_width=True, disabled=is_locked,
                         type="primary" if st.session_state.current_page == Page.PORTFOLIO else "secondary"):
                st.session_state.current_page = Page.PORTFOLIO
                st.rerun()

        st.markdown("---")

        st.markdown("### 📌 自選股專案區")


        if st.session_state.current_page == Page.DASHBOARD:
            with st.form("add_ticker_form", clear_on_submit=True):
                cols = st.columns([3, 1])
                new_ticker = cols[0].text_input("輸入代號", label_visibility="collapsed", placeholder="輸入代號...", disabled=is_locked)
                submitted = cols[1].form_submit_button("新增", disabled=is_locked)

                if submitted and new_ticker:
                    clean_ticker = new_ticker.strip().upper()
                    if not clean_ticker.endswith(".TW") and not clean_ticker.endswith(".TWO"):
                        clean_ticker += ".TW"
                    if clean_ticker not in st.session_state.watch_list:
                        st.session_state.watch_list.append(clean_ticker)
                        save_watchlist(st.session_state.watch_list)
                        if st.session_state.current_ticker is None:
                            on_ticker_change(clean_ticker)
                        st.rerun()

            with st.container(height=250):
                for ticker in st.session_state.watch_list:
                    c1, c2 = st.columns([5, 1])
                    is_current = (ticker == st.session_state.current_ticker)

                    with c1:
                        btn_label = f"🟢 {ticker}" if is_current else ticker
                        # 鎖定狀態或已選中時不可點擊
                        if st.button(btn_label, key=f"btn_{ticker}", use_container_width=True, disabled=is_current or is_locked):
                            on_ticker_change(ticker)
                            st.rerun()
                    with c2:
                        if st.button("x", key=f"del_{ticker}", use_container_width=True, disabled=is_locked):
                            st.session_state.watch_list.remove(ticker)
                            save_watchlist(st.session_state.watch_list)
                            if is_current:
                                if len(st.session_state.watch_list) > 0:
                                    on_ticker_change(st.session_state.watch_list[0])
                                else:
                                    st.session_state.current_ticker = None
                                    st.session_state.ctrl_live = None
                            st.rerun()

            st.markdown("---")

            persona_mapping = {
                "激進型 (AGGRESSIVE)": TradingPersona.AGGRESSIVE,
                "穩健型 (MODERATE)": TradingPersona.MODERATE,
                "保守型 (CONSERVATIVE)": TradingPersona.CONSERVATIVE
            }

            mode_mapping = {
                "波段模式 (SWING)": TradingMode.SWING,
                "當沖模式 (DAY_TRADE)": TradingMode.DAY_TRADE
            }

            saved_persona = st.session_state.user_settings.get("persona", "穩健型 (MODERATE)")
            saved_mode = st.session_state.user_settings.get("mode", "波段模式 (SWING)")
            p_index = list(persona_mapping.keys()).index(saved_persona) if saved_persona in persona_mapping else 1
            m_index = list(mode_mapping.keys()).index(saved_mode) if saved_mode in mode_mapping else 0

            def on_setting_change():
                save_settings(st.session_state.ui_persona, st.session_state.ui_mode)
                st.session_state.user_settings = {"persona": st.session_state.ui_persona, "mode": st.session_state.ui_mode}
                reset_result()

            selected_persona_str = st.selectbox(
                "🧠 戰術性格", list(persona_mapping.keys()), index=p_index,
                key="ui_persona", on_change=on_setting_change, disabled=is_locked
            )

            selected_mode_str = st.selectbox(
                "⚡ 交易模式", list(mode_mapping.keys()), index=m_index,
                key="ui_mode", on_change=on_setting_change, disabled=is_locked
            )

            st.markdown("---")

            if st.button("🚨 重新訓練模型並抓取最新資料 ", type="primary", use_container_width=True, disabled=is_locked or st.session_state.current_ticker is None):
                # 將狀態設為訓練中，並清除現有大腦，觸發 rerun 讓主程式接手訓練
                st.session_state.is_training = True
                st.session_state.ctrl_live = None
                st.session_state.ctrl_bt = None
                reset_result()
                st.rerun()

            return persona_mapping[selected_persona_str], mode_mapping[selected_mode_str]
        else:
            return TradingPersona.MODERATE, TradingMode.SWING
