import time

import streamlit as st

from bt.account import Account
from bt.strategy_config import TradingPersona
from ml.const import TradingMode
from ui.base import is_valid_ticker
from ui.const import Page, SessionKey
from ui.portfolio import load_portfolio, save_portfolio
from ui.state import on_ticker_change, reset_result, save_settings
from ui.stock_names import get_tw_stock_mapping


def render_sidebar() -> tuple[TradingPersona, TradingMode]:
    # 讀取當前的鎖定狀態 (安全存取)
    is_locked = st.session_state.get(SessionKey.IS_TRAINING.value, False) or \
                st.session_state.get(SessionKey.IS_GLOBAL_TRAINING.value, False)

    name_map = get_tw_stock_mapping()

    if SessionKey.PORTFOLIO.value not in st.session_state:
        st.session_state[SessionKey.PORTFOLIO.value] = load_portfolio()
    account: Account = st.session_state[SessionKey.PORTFOLIO.value]

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
            st.markdown("### 📂 操作組合包")

            # 1. 組合包選擇器與刪除按鈕
            sp_names = list(account.sub_portfolios.keys())
            if not sp_names:
                st.warning("請先至「資產管理」建立組合包。")
                return TradingPersona.MODERATE, TradingMode.SWING

            current_sp_name = st.session_state.get("CURRENT_SUB_PORTFOLIO")
            if current_sp_name not in sp_names:
                current_sp_name = sp_names[0]
                st.session_state["CURRENT_SUB_PORTFOLIO"] = current_sp_name

            selected_sp_name = st.selectbox(
                "選擇目前操作的組合",
                sp_names,
                index=sp_names.index(current_sp_name)
            )

            # 如果使用者切換了下拉選單，更新 Session 並重整
            if selected_sp_name != current_sp_name:
                st.session_state["CURRENT_SUB_PORTFOLIO"] = selected_sp_name
                st.session_state[SessionKey.CURRENT_TICKER.value] = None
                st.session_state[SessionKey.CTRL_LIVE.value] = None
                st.rerun()

            current_sp = account.get_sub_portfolio(selected_sp_name)

            # 2. 自選股專案區 (綁定在 current_sp 底下)
            st.markdown("---")
            st.markdown(f"### 📌 【{current_sp.name}】的自選股")

            msg_placeholder = st.empty()

            with st.form("add_ticker_form", clear_on_submit=True):
                cols = st.columns([3, 1])
                new_ticker = cols[0].text_input("輸入代號", label_visibility="collapsed", placeholder="輸入代號 (例: 2330)...", disabled=is_locked)
                submitted = cols[1].form_submit_button("新增", disabled=is_locked)

                if submitted and new_ticker:
                    clean_ticker = new_ticker.strip().upper()
                    if not clean_ticker.endswith(".TW") and not clean_ticker.endswith(".TWO"):
                        clean_ticker += ".TW"

                    # 存取 current_sp.watch_tickers
                    if clean_ticker in current_sp.watch_tickers:
                        msg_placeholder.warning(f"⚠️ {clean_ticker} 已經在此組合包中了！")
                    else:
                        with st.spinner(f"正在驗證 {clean_ticker} 是否存在..."):
                            if is_valid_ticker(clean_ticker):
                                current_sp.watch_tickers.append(clean_ticker)
                                save_portfolio(account) # 存檔！
                                st.session_state[SessionKey.PORTFOLIO.value] = account

                                if st.session_state.get(SessionKey.CURRENT_TICKER.value) is None:
                                    on_ticker_change(clean_ticker)
                                st.rerun()
                            else:
                                msg_placeholder.error(f"❌ 找不到標的 {clean_ticker}！")

            # 3. 渲染該組合包的自選股按鈕
            with st.container(height=250):
                sorted_watch_list = sorted(current_sp.watch_tickers)

                for ticker in sorted_watch_list:
                    c1, c2 = st.columns([5, 1])
                    is_current = (ticker == st.session_state.get(SessionKey.CURRENT_TICKER.value))
                    ch_name = name_map.get(ticker, "")

                    with c1:
                        display_text = f"🟢 {ticker} {ch_name}" if is_current else f"{ticker} {ch_name}"
                        if st.button(display_text, key=f"btn_{selected_sp_name}_{ticker}", use_container_width=True, disabled=is_current or is_locked):
                            on_ticker_change(ticker)
                            st.rerun()
                    with c2:
                        if st.button("x", key=f"del_{selected_sp_name}_{ticker}", use_container_width=True, disabled=is_locked):
                            current_sp.watch_tickers.remove(ticker)
                            save_portfolio(account)
                            st.session_state[SessionKey.PORTFOLIO.value] = account

                            if is_current:
                                if len(current_sp.watch_tickers) > 0:
                                    on_ticker_change(sorted(current_sp.watch_tickers)[0])
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

                # 1. 更新記憶體狀態
                st.session_state[SessionKey.USER_SETTINGS.value] = {"persona": ui_p, "mode": "波段模式 (SWING)"}

                # 2. 呼叫您寫好的 save_settings，把性格存入硬碟 (.json)！
                save_settings(persona=ui_p, mode="波段模式 (SWING)")

                # 3. 清空舊的預測結果
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

@st.dialog("🗑️ 刪除組合包", width="small")
def delete_sub_portfolio_dialog(sp_id: str):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value)
    sp = account.get_sub_portfolio(sp_id)

    st.warning(f"⚠️ 確定要刪除 **【{sp.name}】** 嗎？")
    st.markdown("刪除組合包後，其內部所有的庫存將被強制以**最新收盤價**結算，並將剩餘資金全數退回系統的「未分配流動資金」。")
    st.markdown("此操作**無法復原**！")

    if st.button("🚨 確認刪除並結算", type="primary", use_container_width=True):
        # 1. 結算所有持股市值，並退回資金
        liquidation_value = sp.total_market_value
        returned_cash = sp.allocated_cash + liquidation_value

        # 將退回的資金加到系統總資金
        account.total_cash += returned_cash

        # 2. 從帳戶中移除該組合包
        del account.sub_portfolios[sp_id]

        # 3. 存檔並重整畫面
        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        st.session_state["CURRENT_SUB_PORTFOLIO"] = None # 強制重新選擇
        st.session_state[SessionKey.CURRENT_TICKER.value] = None

        st.toast(f"✅ 已刪除組合包並退回結算資金 ${returned_cash:,.0f} 至總資金。", icon="🗑️")
        time.sleep(1.0)
        st.rerun()


@st.dialog("⚙️ 系統總設定", width="small")
def system_settings_dialog():
    st.markdown("#### 🤖 AI 引擎維運 (MLOps)")
    st.caption("定期讓所有模型吸收最新市場 K 線與趨勢。建議於**每週末**執行一次。過程可能需要數分鐘。")

    if st.button("🔄 執行全域深度重訓", type="primary", use_container_width=True):
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = True
        st.rerun()

    st.divider()

    st.markdown("#### 🧹 帳務清理")
    st.caption("一鍵刪除所有組合包中「已出清 (0 股)」的標的紀錄，保持資產清單乾淨。")
    if st.button("🧹 清除所有已出清標的", use_container_width=True):
        account: Account | None = st.session_state.get(SessionKey.PORTFOLIO.value)
        if not account:
            st.warning("查無帳戶資料。")
            return

        cleaned_count = 0
        # 遍歷所有的子組合包，清理裡面已經賣光 (0股) 的庫存
        for sp in account.sub_portfolios.values():

            # 只要目前 0 股就清除帳務，不管有沒有歷史交易紀錄！
            inactive_tickers = [ticker for ticker, pos in sp.positions.items() if pos.shares == 0]

            for ticker in inactive_tickers:
                del sp.positions[ticker]  # 只從帳務明細中移除
                cleaned_count += 1

        if cleaned_count == 0:
            st.info("目前沒有需要清理的已出清標的！")
        else:
            save_portfolio(account)
            st.session_state[SessionKey.PORTFOLIO.value] = account
            st.toast(f"✅ 成功清除 {cleaned_count} 檔已出清標的紀錄！", icon="🧹")
            time.sleep(0.5)
            st.rerun()

    st.divider()

    st.markdown("#### ⚠️ 危險操作區")
    st.caption("注意：此操作將清空所有的「持股紀錄」，並將資金重置為初始狀態。")
    if st.button("🗑️ 帳戶一鍵清零 (初始化)", type="primary", use_container_width=True):
        empty_account = Account()
        save_portfolio(empty_account)
        st.session_state[SessionKey.PORTFOLIO.value] = empty_account
        st.toast("✅ 帳戶資產與 JSON 存檔已成功初始化！", icon="🗑️")
        time.sleep(0.5)
        st.rerun()
