import streamlit as st

from controller import IDSSController
from ml.engine import QuantAIEngine
from ui.backtest import render_backtest_tab
from ui.chart import render_chart
from ui.const import Page, PortfolioCol
from ui.params import BacktestParams
from ui.portfolio import render_portfolio_page
from ui.report import render_report
from ui.sidebar import render_sidebar
from ui.state import init_session_state


def run_mlops_pipeline(ticker: str):
    """執行完整的雙軌訓練管線，並在訓練前後開關 UI 鎖"""

    with st.status(f"🚀 MLOps 執行中: {ticker}", expanded=True) as status:
        try:
            # 階段 1：更新資料
            st.write("📥 正在從 Yahoo Finance 下載最新歷史與大盤資料...")
            engine_bt = QuantAIEngine(ticker=ticker, oos_days=BacktestParams.MAX_DAYS)
            engine_bt.update_market_data()

            # 階段 2：訓練回測大腦
            st.write(f"🧠 正在訓練回測大腦 (OOS={BacktestParams.MAX_DAYS})... 這將花費數分鐘")
            engine_bt.train_all_models(save_models=True)

            # 階段 3：訓練實盤大腦
            st.write("⚔️ 正在訓練實盤大腦 (OOS=0)...")
            engine_live = QuantAIEngine(ticker=ticker, oos_days=0)
            engine_live.train_all_models(save_models=True)

            status.update(label="✅ 訓練完畢！準備重新載入系統...", state="complete", expanded=False)

            # 解除全域鎖並重整畫面
            st.session_state.is_training = False
            st.rerun()

        except Exception as e:
            status.update(label="❌ 訓練過程中發生錯誤", state="error", expanded=True)
            st.error(f"詳細錯誤: {e}")
            # 發生錯誤時也要記得解鎖
            st.session_state.is_training = False
            if st.button("🔙 返回控制台"):
                st.rerun()
            st.stop()


# ==========================================
# 主程式流 (Main Application Flow)
# ==========================================
def main():
    init_session_state()
    selected_persona, selected_mode = render_sidebar()

    if st.session_state.current_ticker is None:
        st.warning("👈 請先從左側邊欄新增並選擇一檔股票！")
        st.stop()

    # 如果使用者按下了「強制重訓」，狀態會是 True，直接進入訓練管線
    if st.session_state.get('is_training', False):
        run_mlops_pipeline(st.session_state.current_ticker)

    # 確保「實盤大腦」載入
    if st.session_state.ctrl_live is None:
        ctrl = IDSSController(ticker=st.session_state.current_ticker, oos_days=0)

        if ctrl.load_system():
            st.session_state.ctrl_live = ctrl
            st.toast(f"{st.session_state.current_ticker} 實盤引擎就緒！", icon="🟢")
        else:
            # 如果不是因為按按鈕，而是因為真的沒模型，提示使用者啟動管線
            st.warning(f"⚠️ 系統偵測到 **{st.session_state.current_ticker}** 缺乏完整的 AI 模型權重檔。")
            if st.button("🚀 立即啟動 AI 訓練管線", type="primary"):
                st.session_state.is_training = True
                st.rerun()
            st.stop()

    # 取得最新資金與部位資料
    pf = st.session_state.portfolio
    global_cash = pf.get(PortfolioCol.GLOBAL_CASH, 0.0)
    pos_data = pf[PortfolioCol.POSITIONS].get(st.session_state.current_ticker, {PortfolioCol.SHARES: 0, PortfolioCol.AVG_COST: 0.0})
    my_shares = pos_data.get(PortfolioCol.SHARES, 0)
    my_avg_cost = pos_data.get(PortfolioCol.AVG_COST, 0.0)

    # ==========================================
    # 頁面路由系統 (Router)
    # ==========================================
    st.markdown("---")

    if st.session_state.current_page == Page.PORTFOLIO: # ✨ 使用 Enum
        # 進入資產管理頁面
        db_manager = st.session_state.ctrl_live.engine.db if st.session_state.ctrl_live else None
        render_portfolio_page(db_manager)

    else:
        # 進入 IDSS 決策大廳
        if st.session_state.current_ticker is None:
            st.warning("👈 請先從左側邊欄新增並選擇一檔股票！")
            st.stop()

        st.title(f"📊 IDSS 決策大廳 - {st.session_state.current_ticker}")
        st.markdown("---")

        render_chart()

        tab1, tab2 = st.tabs(["🎯 今日行動指令", "⏱️ IDSS 歷史回測模擬"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 產生今日 AI 決策與戰報", type="primary", use_container_width=True, disabled=st.session_state.is_training):
                with st.spinner("神經網路推論中，正在呼叫 Gemini 分析市場新聞空氣..."):
                    # 這裡也要替換掉魔法字串
                    result = st.session_state.ctrl_live.execute_decision(
                        current_cash=global_cash,
                        current_position=my_shares,
                        avg_cost=my_avg_cost,
                        persona=selected_persona,
                        mode=selected_mode
                    )
                    st.session_state.last_result = result

            if st.session_state.last_result is not None:
                render_report(st.session_state.last_result)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            render_backtest_tab(selected_persona)



if __name__ == "__main__":
    st.set_page_config(page_title="IDSS 量化交易終端", page_icon="📈", layout="wide")
    main()
