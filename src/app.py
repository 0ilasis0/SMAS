import time

import streamlit as st

from bt.const import TradeDecision
from controller import IDSSController
from ml.engine import QuantAIEngine
from ui.backtest import render_backtest_tab
from ui.chart import render_chart
from ui.const import APIKey, Page, PortfolioCol, SessionKey
from ui.params import BacktestParams
from ui.portfolio import render_portfolio_page, trade_dialog
from ui.report import render_report
from ui.sidebar import render_sidebar
from ui.state import init_session_state

# 補充一個在 SessionKey 中可能遺漏的自動更新標記
HAS_AUTO_UPDATED_KEY = "has_auto_updated"

def run_mlops_pipeline(ticker: str):
    """執行完整的雙軌訓練管線，並在訓練前後開關 UI 鎖"""

    with st.status(f"🚀 MLOps 執行中: {ticker}", expanded=True) as status:
        try:
            # 階段 1：更新資料
            st.write("📥 正在從 Yahoo Finance 重新下載最新歷史與大盤雙軌資料...")
            engine_bt = QuantAIEngine(ticker=ticker, oos_days=BacktestParams.MAX_DAYS)
            engine_bt.update_market_data(force_wipe=True)

            # 階段 2：訓練回測大腦
            st.write(f"🧠 正在訓練回測大腦 (OOS={BacktestParams.MAX_DAYS})... 這將花費數分鐘")
            engine_bt.train_all_models(save_models=True)

            # 階段 3：訓練實盤大腦
            st.write("⚔️ 正在訓練實盤大腦 (OOS=0)...")
            engine_live = QuantAIEngine(ticker=ticker, oos_days=0)
            engine_live.train_all_models(save_models=True)

            status.update(label="✅ 訓練完畢！準備重新載入系統...", state="complete", expanded=False)

            st.session_state[SessionKey.IS_TRAINING.value] = False
            st.rerun()

        except Exception as e:
            status.update(label="❌ 訓練過程中發生錯誤", state="error", expanded=True)
            st.error(f"詳細錯誤: {e}")
            # 發生錯誤時也要記得解鎖
            st.session_state[SessionKey.IS_TRAINING.value] = False
            if st.button("🔙 返回控制台"):
                st.rerun()
            st.stop()


def run_global_mlops_pipeline():
    """執行全域 MLOps：走訪自選股清單，全部重新抓資料並深度重訓"""
    watch_list = st.session_state.get(SessionKey.WATCH_LIST.value, [])
    total_stocks = len(watch_list)

    if not watch_list:
        st.warning("⚠️ 自選股清單為空，無需訓練。")
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = False
        time.sleep(1)
        st.rerun()
        return

    # 加入整體進度條與動態 Status 標題
    progress_bar = st.progress(0, text="準備啟動 MLOps 全域管線...")

    with st.status("🔄 全域深度重訓執行中 (週末 MLOps 管線)...", expanded=True) as status:
        for idx, ticker in enumerate(watch_list):
            current_step = idx + 1

            # 動態更新進度條文字
            progress_bar.progress(idx / total_stocks, text=f"進度：正在訓練 {ticker} ({current_step}/{total_stocks})")

            # 動態更新狀態框標題
            status.update(label=f"🔄 正在處理 {ticker} ({current_step}/{total_stocks})...")

            try:
                # 階段 1：清除舊資料，下載最新 K 線
                st.write(f"📥 [{ticker}] 清除快取並下載最新資料...")
                engine_bt = QuantAIEngine(ticker=ticker, oos_days=BacktestParams.MAX_DAYS)
                engine_bt.update_market_data(force_wipe=True)

                # 階段 2：訓練回測大腦
                st.write(f"🧠 [{ticker}] 訓練歷史回測大腦...")
                engine_bt.train_all_models(save_models=True)

                # 階段 3：訓練實盤大腦
                st.write(f"⚔️ [{ticker}] 訓練最新實盤大腦...")
                engine_live = QuantAIEngine(ticker=ticker, oos_days=0)
                engine_live.train_all_models(save_models=True)

                st.write(f"✅ [{ticker}] 模型升級完畢！")
            except Exception as e:
                st.error(f"❌ [{ticker}] 訓練失敗: {e}")

        # 完成後將進度條推滿
        progress_bar.progress(1.0, text="🎉 所有模型訓練完畢！")
        status.update(label="🎉 所有自選股模型深度重訓完畢！系統已吸收最新市場趨勢。", state="complete", expanded=False)
        time.sleep(2)

        # 訓練完畢，清空進度條並解鎖重整
        progress_bar.empty()
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = False
        st.rerun()


# ==========================================
# 主程式流 (Main Application Flow)
# ==========================================
def main():
    init_session_state()
    selected_persona, selected_mode = render_sidebar()

    if not st.session_state.get(HAS_AUTO_UPDATED_KEY, False):
        st.toast("🔄 系統首次啟動：正在檢查並同步最新市場收盤資料...", icon="⏳")

        # 1. 建立一個佔位符容器、變數來追蹤是否發生過錯誤
        status_placeholder = st.empty()
        has_error = False
        with status_placeholder.container():
            with st.status("🔄 系統首次啟動：同步所有追蹤標的...", expanded=True) as status:

                # 走訪目前追蹤清單中的所有股票
                watch_list = st.session_state.get(SessionKey.WATCH_LIST.value, [])
                for t in watch_list:
                    st.write(f"📥 正在檢查/同步 {t} 的最新 K 線與大盤資料...")
                    try:
                        temp_engine = QuantAIEngine(ticker=t, oos_days=0)
                        temp_engine.update_market_data()
                    except Exception as e:
                        st.write(f"⚠️ {t} 同步失敗: {e}")
                        has_error = True

                if not has_error:
                    status.update(label="✅ 所有市場資料已校準！", state="complete", expanded=False)
                    st.session_state[HAS_AUTO_UPDATED_KEY] = True
                else:
                    status.update(label="❌ 部分資料同步失敗，請檢查下方訊息", state="error", expanded=True)

                # 標記為已更新，這次瀏覽器開啟期間都不會再觸發
                st.session_state[HAS_AUTO_UPDATED_KEY] = True
                status.update(label="✅ 所有市場資料已校準至最新交易日！", state="complete", expanded=False)

        if not has_error:
            # 稍微停頓 1 秒讓使用者看到「已校準」的狀態
            time.sleep(1)
            status_placeholder.empty()

    current_ticker = st.session_state.get(SessionKey.CURRENT_TICKER.value)

    if current_ticker is None:
        st.warning("👈 請先從左側邊欄新增並選擇一檔股票！")
        st.stop()

    if st.session_state.get(SessionKey.IS_GLOBAL_TRAINING.value, False):
        run_global_mlops_pipeline()
        st.stop()

    # 如果使用者按下了「強制重訓」，狀態會是 True，直接進入訓練管線
    if st.session_state.get(SessionKey.IS_TRAINING.value, False):
        run_mlops_pipeline(current_ticker)
        st.stop()

    # 確保「實盤大腦」載入
    if st.session_state.get(SessionKey.CTRL_LIVE.value) is None:
        ctrl = IDSSController(ticker=current_ticker, oos_days=0)

        if ctrl.load_system():
            st.session_state[SessionKey.CTRL_LIVE.value] = ctrl
            st.toast(f"{current_ticker} 實盤引擎就緒！", icon="🟢")
        else:
            # 如果不是因為按按鈕，而是因為真的沒模型，提示使用者啟動管線
            st.warning(f"⚠️ 系統偵測到 **{current_ticker}** 缺乏完整的 AI 模型權重檔。")
            if st.button("🚀 立即啟動 AI 訓練管線", type="primary"):
                st.session_state[SessionKey.IS_TRAINING.value] = True
                st.rerun()
            st.stop()

    # 取得最新資金與部位資料
    pf = st.session_state.get(SessionKey.PORTFOLIO.value, {})
    global_cash = pf.get(PortfolioCol.GLOBAL_CASH.value, 0.0)
    pos_data = pf.get(PortfolioCol.POSITIONS.value, {}).get(current_ticker, {PortfolioCol.SHARES.value: 0, PortfolioCol.AVG_COST.value: 0.0})
    my_shares = pos_data.get(PortfolioCol.SHARES.value, 0)
    my_avg_cost = pos_data.get(PortfolioCol.AVG_COST.value, 0.0)

    # ==========================================
    # 頁面路由系統 (Router)
    # ==========================================
    st.markdown("---")

    if st.session_state.get(SessionKey.CURRENT_PAGE.value) == Page.PORTFOLIO.value:
        # 渲染下方的既有資產報表
        ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)
        db_manager = ctrl_live.engine.db if ctrl_live else None
        render_portfolio_page(db_manager)

    else:
        # 進入 IDSS 決策大廳
        if current_ticker is None:
            st.warning("👈 請先從左側邊欄新增並選擇一檔股票！")
            st.stop()

        st.title(f"📊 IDSS 決策大廳 - {current_ticker}")
        st.markdown("---")

        render_chart()

        tab1, tab2 = st.tabs(["🎯 今日行動指令", "⏱️ IDSS 歷史回測模擬"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            is_training = st.session_state.get(SessionKey.IS_TRAINING.value, False)
            if st.button("🚀 產生今日 AI 決策與戰報", type="primary", use_container_width=True, disabled=is_training):
                with st.spinner("神經網路推論中，正在呼叫 Gemini 分析市場新聞空氣..."):

                    ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)
                    result = ctrl_live.execute_decision(
                        current_cash=global_cash,
                        current_position=my_shares,
                        avg_cost=my_avg_cost,
                        persona=selected_persona,
                        mode=selected_mode
                    )
                    st.session_state[SessionKey.LAST_RESULT.value] = result

            last_result = st.session_state.get(SessionKey.LAST_RESULT.value)
            if last_result is not None:
                render_report(last_result)

                # ==========================================
                # AI 決策與真實帳房的「一鍵執行橋樑」
                # ==========================================
                decision = last_result.get(APIKey.DECISION.value, {})
                action = decision.get(APIKey.ACTION.value, TradeDecision.HOLD.value)

                # 只有當 AI 建議 BUY 或 SELL 的時候，才顯示執行按鈕
                if action in [TradeDecision.BUY.value, TradeDecision.SELL.value]:
                    st.markdown("---")

                    # 動態改變按鈕顏色與文字
                    btn_icon = "🛒" if action == TradeDecision.BUY.value else "💸"
                    btn_text = f"{btn_icon} 採納 AI 建議，立即執行 {action.upper()} 交易"

                    if st.button(btn_text, type="primary", use_container_width=True, disabled=is_training):
                        # 取得資料庫實例以供彈窗備用
                        ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)
                        db_mgr = ctrl_live.engine.db if ctrl_live else None

                        # 呼叫彈窗，並將 AI 的心血結晶 (動作、價格、股數) 直接灌進去
                        trade_dialog(
                            db_manager=db_mgr,
                            prefill_ticker=current_ticker,
                            prefill_action=action,
                            prefill_price=decision.get(APIKey.TRADE_PRICE.value, 0.0),
                            prefill_shares=decision.get(APIKey.TRADE_SHARES.value, 0)
                        )

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            render_backtest_tab(selected_persona)


if __name__ == "__main__":
    st.set_page_config(page_title="台股量化交易終端", page_icon="📈", layout="wide")
    main()
