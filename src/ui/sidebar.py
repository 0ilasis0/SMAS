import time

import streamlit as st
import yfinance as yf

from bt.strategy_config import TradingPersona
from ml.model.llm_oracle import TradingMode
from ui.const import Page
from ui.portfolio import get_default_portfolio, save_portfolio
from ui.state import (on_ticker_change, reset_result, save_settings,
                      save_watchlist)
from ui.stock_names import get_tw_stock_mapping


def is_valid_ticker(ticker: str) -> bool:
    """輕量級驗證：檢查 Yahoo Finance 是否有該標的之資料"""
    try:
        # 使用 history 抓取 1 天資料，速度最快且不會印出不必要的 log
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1d")
        return not df.empty
    except Exception:
        return False

def render_sidebar() -> tuple[TradingPersona, TradingMode]:
    # 讀取當前的鎖定狀態
    is_locked = st.session_state.get('is_training', False) or st.session_state.get('is_global_training', False)

    name_map = get_tw_stock_mapping()

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

        if st.session_state.current_page == Page.DASHBOARD:
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

                    # 1. 檢查是否已經存在清單中
                    if clean_ticker in st.session_state.watch_list:
                        msg_placeholder.warning(f"⚠️ {clean_ticker} 已經在自選單中了！")

                    # 2. 啟動防呆網路驗證
                    else:
                        with st.spinner(f"正在驗證 {clean_ticker} 是否存在..."):
                            if is_valid_ticker(clean_ticker):
                                # 驗證通過 ➔ 儲存並重整
                                st.session_state.watch_list.append(clean_ticker)
                                save_watchlist(st.session_state.watch_list)
                                if st.session_state.current_ticker is None:
                                    on_ticker_change(clean_ticker)
                                st.rerun()
                            else:
                                # 驗證失敗 ➔ 拒絕儲存，並報錯
                                msg_placeholder.error(f"❌ 找不到標的 {clean_ticker}！請確認代號是否正確。")

            with st.container(height=250):
                for ticker in st.session_state.watch_list:
                    c1, c2 = st.columns([5, 1])
                    is_current = (ticker == st.session_state.current_ticker)

                    # 從字典中查出中文名稱，查不到就留白
                    ch_name = name_map.get(ticker, "")

                    with c1:
                        display_text = f"🟢 {ticker} {ch_name}" if is_current else f"{ticker} {ch_name}"
                        if st.button(display_text, key=f"btn_{ticker}", use_container_width=True, disabled=is_current or is_locked):
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

        st.divider()

        # 全域系統總設定
        if st.button("⚙️ 系統總設定", use_container_width=True):
            system_settings_dialog()

        if st.session_state.current_page == Page.DASHBOARD:
            return persona_mapping[selected_persona_str], mode_mapping[selected_mode_str]
        else:
            return TradingPersona.MODERATE, TradingMode.SWING


@st.dialog("⚙️ 系統總設定", width="small")
def system_settings_dialog():
    st.markdown("#### 🤖 AI 引擎維運 (MLOps)")
    st.caption("定期讓所有模型吸收最新市場 K 線與趨勢。建議於**每週末**執行一次。過程可能需要數分鐘。")

    if st.button("🔄 執行全域深度重訓", type="primary", use_container_width=True):
        st.session_state.is_global_training = True
        st.rerun()

    st.divider()

    st.markdown("#### ⚠️ 危險操作區")
    st.caption("注意：此操作將清空所有的「持股紀錄」，並將資金重置為初始狀態。")
    if st.button("🗑️ 帳戶一鍵清零 (初始化)", type="primary", use_container_width=True):
        # 確保這裡有 import get_default_portfolio 和 save_portfolio
        from ui.portfolio import get_default_portfolio, save_portfolio

        clean_portfolio = get_default_portfolio()
        st.session_state.portfolio = clean_portfolio
        save_portfolio(clean_portfolio) # 同步抹除 JSON 檔案
        st.toast("✅ 帳戶資產與 JSON 存檔已成功初始化！", icon="🗑️")
        time.sleep(0.5)
        st.rerun()
