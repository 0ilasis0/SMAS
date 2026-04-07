import time

import streamlit as st

from bt.strategy_config import TradingPersona
from ml.const import TradingMode
from ui.base import is_valid_ticker
from ui.const import Page, SessionKey
from ui.state import (on_ticker_change, reset_result, save_settings,
                      save_watchlist)
from ui.stock_names import get_tw_stock_mapping


def render_sidebar() -> tuple[TradingPersona, TradingMode]:
    # 讀取當前的鎖定狀態 (安全存取)
    is_locked = st.session_state.get(SessionKey.IS_TRAINING.value, False) or \
                st.session_state.get(SessionKey.IS_GLOBAL_TRAINING.value, False)

    name_map = get_tw_stock_mapping()

    with st.sidebar:
        st.title("⚙️ IDSS 控制台")

        if is_locked:
            st.warning("⏳ 系統正在進行 AI 模型訓練，控制台暫時鎖定。")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈 決策大廳", use_container_width=True, disabled=is_locked,
                         type="primary" if st.session_state.get(SessionKey.CURRENT_PAGE.value) == Page.DASHBOARD.value else "secondary"):
                st.session_state[SessionKey.CURRENT_PAGE.value] = Page.DASHBOARD.value
                st.rerun()
        with c2:
            if st.button("💼 資產管理", use_container_width=True, disabled=is_locked,
                         type="primary" if st.session_state.get(SessionKey.CURRENT_PAGE.value) == Page.PORTFOLIO.value else "secondary"):
                st.session_state[SessionKey.CURRENT_PAGE.value] = Page.PORTFOLIO.value
                st.rerun()

        if st.session_state.get(SessionKey.CURRENT_PAGE.value) == Page.DASHBOARD.value:
            st.markdown("---")
            st.markdown("### 📌 自選股專案區")

            # 建立一個佔位符，用來顯示錯誤訊息 (放在 form 外面才不會被 clear_on_submit 清掉)
            msg_placeholder = st.empty()

            with st.form("add_ticker_form", clear_on_submit=True):
                cols = st.columns([3, 1])
                new_ticker = cols[0].text_input("輸入代號", label_visibility="collapsed", placeholder="輸入代號 (例: 2330)...", disabled=is_locked)
                submitted = cols[1].form_submit_button("新增", disabled=is_locked)

                if submitted and new_ticker:
                    clean_ticker = new_ticker.strip().upper()

                    # 自動補齊台股後綴
                    if not clean_ticker.endswith(".TW") and not clean_ticker.endswith(".TWO"):
                        clean_ticker += ".TW"

                    watch_list = st.session_state.get(SessionKey.WATCH_LIST.value, [])
                    # 1. 檢查是否已經存在清單中
                    if clean_ticker in watch_list:
                        msg_placeholder.warning(f"⚠️ {clean_ticker} 已經在自選單中了！")

                    # 2. 啟動防呆網路驗證
                    else:
                        with st.spinner(f"正在驗證 {clean_ticker} 是否存在..."):
                            if is_valid_ticker(clean_ticker):
                                # 驗證通過 ➔ 儲存並重整
                                watch_list.append(clean_ticker)
                                st.session_state[SessionKey.WATCH_LIST.value] = watch_list
                                save_watchlist(watch_list)
                                if st.session_state.get(SessionKey.CURRENT_TICKER.value) is None:
                                    on_ticker_change(clean_ticker)
                                st.rerun()
                            else:
                                # 驗證失敗 ➔ 拒絕儲存，並報錯
                                msg_placeholder.error(f"❌ 找不到標的 {clean_ticker}！請確認代號是否正確。")

            with st.container(height=250):
                watch_list = st.session_state.get(SessionKey.WATCH_LIST.value, [])
                for ticker in watch_list:
                    c1, c2 = st.columns([5, 1])
                    is_current = (ticker == st.session_state.get(SessionKey.CURRENT_TICKER.value))

                    # 從字典中查出中文名稱，查不到就留白
                    ch_name = name_map.get(ticker, "")

                    with c1:
                        display_text = f"🟢 {ticker} {ch_name}" if is_current else f"{ticker} {ch_name}"
                        if st.button(display_text, key=f"btn_{ticker}", use_container_width=True, disabled=is_current or is_locked):
                            on_ticker_change(ticker)
                            st.rerun()
                    with c2:
                        if st.button("x", key=f"del_{ticker}", use_container_width=True, disabled=is_locked):
                            watch_list.remove(ticker)
                            st.session_state[SessionKey.WATCH_LIST.value] = watch_list
                            save_watchlist(watch_list)
                            if is_current:
                                if len(watch_list) > 0:
                                    on_ticker_change(watch_list[0])
                                else:
                                    st.session_state[SessionKey.CURRENT_TICKER.value] = None
                                    st.session_state[SessionKey.CTRL_LIVE.value] = None
                            st.rerun()

            st.markdown("---")

            persona_mapping = {
                "激進型 (AGGRESSIVE)": TradingPersona.AGGRESSIVE,
                "穩健型 (MODERATE)": TradingPersona.MODERATE,
                "保守型 (CONSERVATIVE)": TradingPersona.CONSERVATIVE
            }

            user_settings = st.session_state.get(SessionKey.USER_SETTINGS.value, {})
            saved_persona = user_settings.get("persona", "穩健型 (MODERATE)")
            p_index = list(persona_mapping.keys()).index(saved_persona) if saved_persona in persona_mapping else 1

            def on_setting_change():
                ui_p = st.session_state.get(SessionKey.UI_PERSONA.value)
                st.session_state[SessionKey.USER_SETTINGS.value] = {"persona": ui_p}
                reset_result()

            selected_persona_str = st.selectbox(
                "🧠 戰術性格", list(persona_mapping.keys()), index=p_index,
                key=SessionKey.UI_PERSONA.value, on_change=on_setting_change, disabled=is_locked
            )

            st.markdown("---")

            if st.button("🚨 重新訓練模型並抓取最新資料 ", type="primary", use_container_width=True, disabled=is_locked or st.session_state.get(SessionKey.CURRENT_TICKER.value) is None):
                # 將狀態設為訓練中，並清除現有大腦，觸發 rerun 讓主程式接手訓練
                st.session_state[SessionKey.IS_TRAINING.value] = True
                st.session_state[SessionKey.CTRL_LIVE.value] = None
                st.session_state[SessionKey.CTRL_BT.value] = None
                reset_result()
                st.rerun()

        st.divider()

        # 全域系統總設定
        if st.button("⚙️ 系統總設定", use_container_width=True):
            system_settings_dialog()

        if st.session_state.get(SessionKey.CURRENT_PAGE.value) == Page.DASHBOARD.value:
            return persona_mapping[selected_persona_str]
        else:
            return TradingPersona.MODERATE


@st.dialog("⚙️ 系統總設定", width="small")
def system_settings_dialog():
    st.markdown("#### 🤖 AI 引擎維運 (MLOps)")
    st.caption("定期讓所有模型吸收最新市場 K 線與趨勢。建議於**每週末**執行一次。過程可能需要數分鐘。")

    if st.button("🔄 執行全域深度重訓", type="primary", use_container_width=True):
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = True
        st.rerun()

    st.divider()

    st.markdown("#### ⚠️ 危險操作區")
    st.caption("注意：此操作將清空所有的「持股紀錄」，並將資金重置為初始狀態。")
    if st.button("🗑️ 帳戶一鍵清零 (初始化)", type="primary", use_container_width=True):
        # 確保這裡有 import get_default_portfolio 和 save_portfolio
        from ui.portfolio import get_default_portfolio, save_portfolio

        clean_portfolio = get_default_portfolio()
        st.session_state[SessionKey.PORTFOLIO.value] = clean_portfolio
        save_portfolio(clean_portfolio) # 同步抹除 JSON 檔案
        st.toast("✅ 帳戶資產與 JSON 存檔已成功初始化！", icon="🗑️")
        time.sleep(0.5)
        st.rerun()