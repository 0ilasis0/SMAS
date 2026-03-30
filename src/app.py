import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bt.strategy_config import TradingPersona
from controller import IDSSController
from ml.model.llm_oracle import TradingMode


# ==========================================
# 0. 回呼函數 (Callbacks)
# ==========================================
def reset_result():
    st.session_state.last_result = None

def on_ticker_change():
    st.session_state.current_ticker = st.session_state.new_ticker
    st.session_state.controller = None
    reset_result()

# ==========================================
# 1. UI 元件模組化 (View Layer)
# ==========================================
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
            st.session_state.controller = None
            reset_result()
            st.rerun()

        return persona_mapping[selected_persona_str], mode_mapping[selected_mode_str]

def render_chart():
    """渲染中央 K 線圖"""
    with st.expander("📉 近期走勢圖 (近 60 日)", expanded=True):
        try:
            ctrl = st.session_state.controller
            df_recent = ctrl.engine.db.get_aligned_market_data(st.session_state.current_ticker, []).tail(60)
            if not df_recent.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_recent.index, open=df_recent['Open'], high=df_recent['High'], low=df_recent['Low'], close=df_recent['Close'], increasing_line_color='red', decreasing_line_color='green', name='K線'))

                ma5 = df_recent['Close'].rolling(window=5).mean().bfill()
                ma20 = df_recent['Close'].rolling(window=20).mean().bfill()

                fig.add_trace(go.Scatter(x=df_recent.index, y=ma5, line=dict(color='orange', width=1.5), name='5MA'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma20, line=dict(color='purple', width=1.5), name='20MA'))

                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"無法渲染 K 線圖: {e}")

def render_report(result: dict):
    """將戰報渲染邏輯獨立，保持主程式乾淨"""
    st.success(f"決策生成完畢！(目標日期: {result.get('date', '未知')})")

    action = result["decision"]["action"]
    color = "#ff4b4b" if action == "BUY" else "#00cc66" if action == "SELL" else "#a6a6a6"
    icon = "🔥" if action == "BUY" else "🩸" if action == "SELL" else "🛡️"

    st.markdown(f"<h2 style='text-align: center; color: {color};'>{icon} 終極指令：{action}</h2>", unsafe_allow_html=True)

    if action != "HOLD":
        shares = result['decision']['trade_shares']
        price = result['decision']['trade_price']
        st.markdown(f"<p style='text-align: center; font-size: 18px;'>建議股數：<b>{shares:,}</b> 股 | 預估觸價：<b>{price:,.2f}</b> 元</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align: center; font-size: 18px; color: gray;'>無觸發交易訊號，維持既有資金與持股配置</p>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 AI 雙腦與大盤雷達指標")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("綜合決策勝率 (Meta)", f"{result['ai_signals']['final_prob']:.2%}")
    m2.metric("技術面動能 (XGB)", f"{result['ai_signals']['xgb_prob']:.2%}")
    m3.metric("K線型態辨識 (DL)", f"{result['ai_signals']['dl_prob']:.2%}")
    m4.metric("大盤環境安全度", f"{result['ai_signals']['market_safe']:.2%}")

    st.markdown("### 📰 Gemini 新聞")
    score = result['sentiment']['score']
    sentiment_color = "red" if score >= 7 else "green" if score <= 3 else "orange"
    st.info(f"**情緒分數：:{sentiment_color}[{score} / 10]** \n\n**判讀理由：** {result['sentiment']['reason']}")

    st.markdown("### 🤖 總裁戰報")
    st.warning(result["report"], icon="💡")

    with st.expander("🛠️ 展開系統底層 JSON 數據"):
        st.json(result)

# ==========================================
# 2. 主程式流 (Main Application Flow)
# ==========================================
def main():
    # 初始化狀態
    if 'watch_list' not in st.session_state:
        st.session_state.watch_list = ["2330.TW", "2388.TW", "5469.TW", "2337.TW"]
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = st.session_state.watch_list[0]
    if 'controller' not in st.session_state:
        st.session_state.controller = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # 渲染側邊欄並取得設定
    selected_persona, selected_mode = render_sidebar()

    # 確保引擎載入
    if st.session_state.controller is None:
        with st.spinner(f"正在喚醒 {st.session_state.current_ticker} 的 AI 模型與神經網路權重..."):
            ctrl = IDSSController(ticker=st.session_state.current_ticker)
            if ctrl.load_system():
                st.session_state.controller = ctrl
                st.toast(f"{st.session_state.current_ticker} 引擎上線就緒！", icon="🟢")
            else:
                st.error("❌ 模型載入失敗，請確認該標的已經過訓練或資料完整。")
                st.stop()

    # 主畫面
    st.title(f"📊 IDSS 決策大廳 - {st.session_state.current_ticker}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 總淨值 (預覽)", "2,000,000")
    c2.metric("💵 可用現金 (預覽)", "2,000,000")
    c3.metric("📂 持股市值 (預覽)", "0")
    st.markdown("---")

    render_chart()

    # 執行按鈕
    if st.button("🚀 產生今日 AI 決策與戰報", type="primary", use_container_width=True):
        with st.spinner("神經網路推論中，正在呼叫 Gemini 分析市場新聞空氣..."):
            result = st.session_state.controller.execute_decision(
                current_cash=2000000.0,
                current_position=0,
                avg_cost=0.0,
                persona=selected_persona,
                mode=selected_mode
            )
            st.session_state.last_result = result

    # 渲染戰報
    if st.session_state.last_result is not None:
        render_report(st.session_state.last_result)

if __name__ == "__main__":
    st.set_page_config(page_title="IDSS 量化交易終端", page_icon="📈", layout="wide")
    main()
