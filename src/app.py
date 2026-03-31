import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bt.const import DecisionAction
from bt.strategy_config import TradingPersona
from controller import IDSSController
from data.const import StockCol
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

@st.cache_data(ttl=3600)
def get_cached_market_data(ticker: str):
    # 這裡必須在函數內實例化或傳入 db，避免 thread 問題
    from data.manager import DataManager
    db = DataManager()
    return db.get_aligned_market_data(ticker, []).tail(60)

def render_chart():
    """渲染中央 K 線圖 (支援動態縮放、全中文月份、無縫接合斷點)"""
    with st.expander("📉 近期走勢圖 (預設顯示近 1 年，可自由縮放檢視歷史)", expanded=True):
        try:
            ctrl = st.session_state.controller

            # 抓取近 3 年的資料，讓年線算得出來，且有足夠歷史可回顧
            df_recent = ctrl.engine.db.get_aligned_market_data(st.session_state.current_ticker, []).tail(720)

            if not df_recent.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_recent.index,
                                             open=df_recent[StockCol.OPEN],
                                             high=df_recent[StockCol.HIGH],
                                             low=df_recent[StockCol.LOW],
                                             close=df_recent[StockCol.CLOSE],
                                             increasing_line_color='red',
                                             decreasing_line_color='green',
                                             name='K線'))

                # 計算所有均線
                ma5 = df_recent[StockCol.CLOSE].rolling(window=5).mean().bfill()
                ma20 = df_recent[StockCol.CLOSE].rolling(window=20).mean().bfill()
                ma60 = df_recent[StockCol.CLOSE].rolling(window=60).mean().bfill()
                # 🚀 新增 2：加入 240MA (年線)
                ma240 = df_recent[StockCol.CLOSE].rolling(window=240).mean().bfill()

                fig.add_trace(go.Scatter(x=df_recent.index, y=ma5, line=dict(color='orange', width=1.5), name='5MA(週線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma20, line=dict(color='purple', width=1.5), name='20MA(月線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma60, line=dict(color='blue', width=1.5), name='60MA(季線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma240, line=dict(color='black', width=1.5), name='240MA(年線)'))

                # 建立從第一天到最後一天的連續日曆
                all_dates = pd.date_range(start=df_recent.index.min(), end=df_recent.index.max())
                # 找出 DataFrame 中沒有的日期 (週末 + 國定假日)
                missing_dates = all_dates.difference(df_recent.index)
                missing_dates_str = missing_dates.strftime('%Y-%m-%d').tolist()

                # 找出每個月的第一個交易日來放置標籤
                first_days_of_month = df_recent.groupby([df_recent.index.year, df_recent.index.month]).head(1)
                tickvals = first_days_of_month.index
                # 如果是 1 月，加上年份 (如 2026年1月)，其餘只顯示月份 (如 2月、3月)
                ticktext = [f"{d.year}年{d.month}月" if d.month == 1 else f"{d.month}月" for d in tickvals]

                # 計算初始畫面要顯示的範圍 (最近 240 天)
                initial_start_date = df_recent.index[-240] if len(df_recent) >= 240 else df_recent.index[0]
                end_date = df_recent.index[-1]

                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    xaxis_rangeslider_visible=True,

                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.6)"
                    ),

                    xaxis=dict(
                        # 抽掉所有非交易日，完美接合 K 線
                        rangebreaks=[dict(values=missing_dates_str)],

                        # 右側邊界精準鎖定在最後一個交易日，不會顯示未來的空白
                        range=[initial_start_date, end_date],

                        # 套用中文月份標籤
                        tickmode='array',
                        tickvals=tickvals,
                        ticktext=ticktext,
                        showgrid=True,
                        gridcolor="rgba(200, 200, 200, 0.2)"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        'scrollZoom': True,       # 允許滑鼠滾輪縮放
                        'displayModeBar': False,  # 隱藏右上角所有小工具 (畫圖、刪除等)，回歸極簡
                        'displaylogo': False
                    }
                )
        except Exception as e:
            st.caption(f"無法渲染 K 線圖: {e}")

def render_report(result: dict):
    """將戰報渲染邏輯獨立，保持主程式乾淨"""
    if result.get("status") != "success":
        st.error(f"❌ 發生錯誤: {result.get('message', '未知錯誤')}")
        return

    st.success(f"決策生成完畢！(目標日期: {result.get('date', '未知')})")

    action = result["decision"]["action"]

    action_style_map = {
        DecisionAction.BUY:  {"label": "買進", "color": "#ff4b4b", "icon": "🔥"},
        DecisionAction.SELL: {"label": "賣出", "color": "#00cc66", "icon": "🩸"},
        DecisionAction.HOLD: {"label": "觀望", "color": "#a6a6a6", "icon": "🛡️"}
    }

    style = action_style_map.get(action, {"label": "未知操作", "color": "#a6a6a6", "icon": "❓"})

    # 渲染主標題
    st.markdown(
        f"<h2 style='text-align: center; color: {style['color']};'>"
        f"{style['icon']} 建議執行：{style['label']}</h2>",
        unsafe_allow_html=True
    )

    # 副標題：判斷是否需要顯示交易細節
    if action != "HOLD":
        shares = result['decision']['trade_shares']
        price = result['decision']['trade_price']
        st.markdown(
            f"<p style='text-align: center; font-size: 18px;'>"
            f"建議股數：<b>{shares:,}</b> 股 | 預估觸價：<b>{price:,.2f}</b> 元</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='text-align: center; font-size: 18px; color: gray;'>"
            f"無觸發交易訊號，維持既有資金與持股配置</p>",
            unsafe_allow_html=True
        )

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

# ==========================================
# 2. 主程式流 (Main Application Flow)
# ==========================================
def main():
    # 初始化狀態
    if 'watch_list' not in st.session_state:
        st.session_state.watch_list = ["00631L.TW", "2330.TW", "2388.TW", "5469.TW", "2337.TW"]
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
