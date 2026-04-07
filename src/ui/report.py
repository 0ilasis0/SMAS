import streamlit as st

from bt.const import TradeDecision
from const import Color
from ml.const import OracleCol, QuoteCol, SignalCol
from ui.const import APIKey


def render_report(result: dict):
    """將戰報渲染邏輯獨立，保持主程式乾淨"""

    if result.get(APIKey.STATUS.value) != "success":
        st.error(f"❌ 發生錯誤: {result.get(APIKey.MESSAGE.value, '未知錯誤')}")
        return

    st.success(f"決策生成完畢！(正在預測目標日期: {result.get(QuoteCol.DATE.value, '未知')})")

    action = result[APIKey.DECISION.value][APIKey.ACTION.value]

    action_style_map = {
        TradeDecision.BUY.value:  {"label": "買進", "color": Color.RED.value, "icon": "🔥"},
        TradeDecision.SELL.value: {"label": "賣出", "color": Color.GREEN.value, "icon": "🩸"},
        TradeDecision.HOLD.value: {"label": "觀望", "color": Color.GRAY.value, "icon": "🛡️"}
    }

    style = action_style_map.get(action, {"label": "未知操作", "color": Color.GRAY.value, "icon": "❓"})

    st.markdown(
        f"<h2 style='text-align: center; color: {style['color']};'>{style['icon']} 建議執行：{style['label']}</h2>",
        unsafe_allow_html=True
    )

    if action != TradeDecision.HOLD.value:
        shares = result[APIKey.DECISION.value][APIKey.TRADE_SHARES.value]
        price = result[APIKey.DECISION.value][APIKey.TRADE_PRICE.value]
        st.markdown(
            f"<p style='text-align: center; font-size: 18px;'>建議股數：<b>{shares:,}</b> 股 | 預估觸價：<b>{price:,.2f}</b> 元</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p style='text-align: center; font-size: 18px; color: {Color.GRAY.value};'>無觸發交易訊號，維持既有資金與持股配置</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 📈 AI 雙腦與大盤雷達指標")
    m1, m2, m3, m4 = st.columns(4)
    ai_sigs = result[APIKey.AI_SIGNALS.value]
    m1.metric("綜合決策勝率 (Meta)", f"{ai_sigs[SignalCol.PROB_FINAL.value]:.2%}")
    m2.metric("技術面動能 (XGB)", f"{ai_sigs[SignalCol.PROB_XGB.value]:.2%}")
    m3.metric("K線型態辨識 (DL)", f"{ai_sigs[SignalCol.PROB_DL.value]:.2%}")
    m4.metric("大盤環境安全度", f"{ai_sigs[SignalCol.PROB_MARKET_SAFE.value]:.2%}")

    st.markdown("### 📰 Gemini 新聞情緒")
    sentiment = result[APIKey.SENTIMENT.value]
    score = sentiment[OracleCol.SCORE.value]

    sentiment_color = Color.RED.value if score >= 7 else Color.GREEN.value if score <= 3 else Color.ORANGE.value
    st.info(f"**情緒分數：:{sentiment_color}[{score} / 10]** \n\n**判讀理由：** {sentiment[OracleCol.REASON.value]}")

    st.markdown("### 🤖 總裁戰報")
    st.warning(result[APIKey.REPORT.value], icon="💡")
