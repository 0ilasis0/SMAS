import streamlit as st

from bt.const import DecisionAction
from const import Color
from ml.const import MarketCol


def render_report(result: dict):
    """將戰報渲染邏輯獨立，保持主程式乾淨"""
    if result.get("status") != "success":
        st.error(f"❌ 發生錯誤: {result.get('message', '未知錯誤')}")
        return

    st.success(f"決策生成完畢！(正在預測目標日期: {result.get('date', '未知')})")

    action = result["decision"]["action"]

    action_style_map = {
        DecisionAction.BUY:  {"label": "買進", "color": "#ff4b4b", "icon": "🔥"},
        DecisionAction.SELL: {"label": "賣出", "color": "#00cc66", "icon": "🩸"},
        DecisionAction.HOLD: {"label": "觀望", "color": "#a6a6a6", "icon": "🛡️"}
    }

    style = action_style_map.get(action, {"label": "未知操作", "color": "#a6a6a6", "icon": "❓"})

    st.markdown(
        f"<h2 style='text-align: center; color: {style['color']};'>{style['icon']} 建議執行：{style['label']}</h2>",
        unsafe_allow_html=True
    )

    if action != "HOLD":
        shares = result['decision']['trade_shares']
        price = result['decision']['trade_price']
        st.markdown(
            f"<p style='text-align: center; font-size: 18px;'>建議股數：<b>{shares:,}</b> 股 | 預估觸價：<b>{price:,.2f}</b> 元</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='text-align: center; font-size: 18px; color: gray;'>無觸發交易訊號，維持既有資金與持股配置</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📈 AI 雙腦與大盤雷達指標")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("綜合決策勝率 (Meta)", f"{result['ai_signals'][MarketCol.PROB_FINAL]:.2%}")
    m2.metric("技術面動能 (XGB)", f"{result['ai_signals'][MarketCol.PROB_XGB]:.2%}")
    m3.metric("K線型態辨識 (DL)", f"{result['ai_signals'][MarketCol.PROB_DL]:.2%}")
    m4.metric("大盤環境安全度", f"{result['ai_signals'][MarketCol.PROB_MARKET_SAFE]:.2%}")

    st.markdown("### 📰 Gemini 新聞情緒")
    score = result['sentiment'][MarketCol.SENTIMENT_SCORE]
    sentiment_color = Color.RED if score >= 7 else Color.GREEN if score <= 3 else Color.ORANGE
    st.info(f"**情緒分數：:{sentiment_color}[{score} / 10]** \n\n**判讀理由：** {result['sentiment'][MarketCol.SENTIMENT_REASON]}")

    st.markdown("### 🤖 總裁戰報")
    st.warning(result["report"], icon="💡")