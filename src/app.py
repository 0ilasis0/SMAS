import streamlit as st

from controller import IDSSController
from ui.backtest import render_backtest_tab
from ui.chart import render_chart
from ui.report import render_report
from ui.sidebar import render_sidebar
# 引入 UI 拆分模組
from ui.state import init_session_state


# ==========================================
# 主程式流 (Main Application Flow)
# ==========================================
def main():
    # 1. 初始化狀態
    init_session_state()

    # 2. 渲染側邊欄並取得設定
    selected_persona, selected_mode = render_sidebar()

    # 3. 確保「實盤大腦 (OOS=0)」載入
    if st.session_state.ctrl_live is None:
        with st.spinner(f"正在喚醒 {st.session_state.current_ticker} 的最新實盤模型 (OOS=0)..."):
            ctrl = IDSSController(ticker=st.session_state.current_ticker, oos_days=0)
            if ctrl.load_system():
                st.session_state.ctrl_live = ctrl
                st.toast(f"{st.session_state.current_ticker} 實盤引擎就緒！", icon="🟢")
            else:
                st.error("❌ 模型載入失敗，請確認該標的已經過訓練或資料完整。")
                st.stop()

    # 4. 主畫面排版
    st.title(f"📊 IDSS 決策大廳 - {st.session_state.current_ticker}")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 總淨值 (預覽)", "2,000,000")
    c2.metric("💵 可用現金 (預覽)", "2,000,000")
    c3.metric("📂 持股市值 (預覽)", "0")
    st.markdown("---")

    # 渲染圖表
    render_chart()

    # 5. 分頁渲染
    tab1, tab2 = st.tabs(["🎯 今日行動指令", "⏱️ IDSS 歷史回測模擬"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 產生今日 AI 決策與戰報", type="primary", use_container_width=True):
            with st.spinner("神經網路推論中，正在呼叫 Gemini 分析市場新聞空氣..."):
                result = st.session_state.ctrl_live.execute_decision(
                    current_cash=2000000.0,
                    current_position=0,
                    avg_cost=0.0,
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
