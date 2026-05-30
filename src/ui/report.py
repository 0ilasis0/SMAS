from datetime import datetime

import pytz
import streamlit as st

from bt.const import TradeDecision
from const import Color, StatusCol
from ml.const import MetricsCol, OracleCol, QuoteCol, SignalCol
from ml.params import MLDefault
from ui.const import APIKey


def render_report(result: dict):
    """將財報渲染邏輯獨立，保持主程式乾淨"""

    if result.get(APIKey.STATUS.value) != StatusCol.SUCCESS:
        st.error(f"❌ 發生錯誤: {result.get(APIKey.MESSAGE.value, '未知錯誤')}")
        return

    target_date_str = result.get(QuoteCol.DATE.value, '未知')

    # --- 智慧判斷盤中或收盤狀態 ---
    status_text = f"以 {target_date_str} 收盤資料為基準，預測下個交易日走勢"

    if target_date_str != '未知':
        try:
            # 取得目前的台灣時間
            tw_tz = pytz.timezone('Asia/Taipei')
            now_tw = datetime.now(tw_tz)
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

            # 如果財報日期就是「今天」
            if target_date == now_tw.date():
                market_open = now_tw.replace(hour=9, minute=0, second=0, microsecond=0)
                market_close = now_tw.replace(hour=13, minute=30, second=0, microsecond=0)

                # 判斷是否正處於盤中 (09:00 ~ 13:30)
                if market_open <= now_tw <= market_close:
                    status_text = f"以 {target_date_str} **盤中即時資料**為基準，動態評估今日走勢 (⚠️ 提示：盤中 K 線尚未定型易生雜訊，建議於 13:00 後產出最終財報)"
                elif now_tw > market_close:
                    status_text = f"以 {target_date_str} **收盤資料**為基準，預測下個交易日走勢"
        except Exception:
            pass

    # 渲染頂部狀態列
    st.success(f"決策生成完畢！👉 {status_text}")

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

    metrics = result.get(MetricsCol.DICT_KEY.value, {})
    xgb_auc = metrics.get(MetricsCol.XGB_AUC, MLDefault.FALLBACK_AUC)
    dl_auc = metrics.get(MetricsCol.DL_AUC, MLDefault.FALLBACK_AUC)
    market_auc = metrics.get(MetricsCol.MARKET_AUC, MLDefault.FALLBACK_AUC)
    market_thresh = metrics.get(MetricsCol.MARKET_THRESH, MLDefault.FALLBACK_THRESH)

    def get_health_icon(auc):
        """根據 AUC 回傳簡化的燈號圖示"""
        if auc >= MLDefault.HIGH_AUC: return "🟢"
        elif auc >= MLDefault.MID_AUC: return "🟡"
        else: return "🔴"

    # 建立一個客製化的 HTML 渲染器，完美模擬 st.metric，並把 icon 變為右上角的上標 (sup)
    def render_custom_metric(label, value_str, icon=""):
        return f"""
        <div style="display: flex; flex-direction: column; margin-bottom: 1rem;">
            <span style="font-size: 14px; color: #7f8c8d; font-family: sans-serif;">{label}</span>
            <span style="font-size: 2.25rem; font-weight: 600; line-height: 1.2; font-family: sans-serif;">
                {value_str}<sup style="font-size: 1.2rem; margin-left: 2px;">{icon}</sup>
            </span>
        </div>
        """

    m1, m2, m3, m4 = st.columns(4)
    ai_sigs = result[APIKey.AI_SIGNALS.value]

    # 使用 custom HTML 取代原生的 st.metric
    m1.markdown(
        render_custom_metric("綜合決策勝率 (Meta)", f"{ai_sigs[SignalCol.PROB_FINAL.value]:.2%}"),
        unsafe_allow_html=True
    )
    m2.markdown(
        render_custom_metric("技術面動能 (XGB)", f"{ai_sigs[SignalCol.PROB_XGB.value]:.2%}", get_health_icon(xgb_auc)),
        unsafe_allow_html=True
    )
    m3.markdown(
        render_custom_metric("K線型態辨識 (DL)", f"{ai_sigs[SignalCol.PROB_DL.value]:.2%}", get_health_icon(dl_auc)),
        unsafe_allow_html=True
    )
    m4.markdown(
        render_custom_metric("大盤環境安全度", f"{ai_sigs[SignalCol.PROB_MARKET_SAFE.value]:.2%}", get_health_icon(market_auc)),
        unsafe_allow_html=True
    )

    # AI 引擎底層健康度：收合成齒輪圖示
    if metrics:
        def get_health_status(auc):
            if auc >= MLDefault.HIGH_AUC: return "🟢 **極佳** (具備強大預測優勢)"
            elif auc >= MLDefault.MID_AUC: return "🟡 **正常** (穩定運作中)"
            else: return "🔴 **重新適應中** (建議過幾天後執行重新訓練)"

        with st.popover("⚙️", help="AI 引擎底層健康度 (點擊查看進階指標)"):
            st.markdown(f"""
            - **大盤防禦雷達 (Market Brain):** 狀態 {get_health_status(market_auc)}
              - 交叉驗證 AUC: `{market_auc:.4f}`
              - 實戰防禦警戒線: `{market_thresh:.4f}`
            - **技術面動能 (XGBoost):** 狀態 {get_health_status(xgb_auc)}
              - 交叉驗證 AUC: `{xgb_auc:.4f}`
            - **K線型態辨識 (Deep Learning):** 狀態 {get_health_status(dl_auc)}
              - 交叉驗證 AUC: `{dl_auc:.4f}`

            *(💡 溫馨提示：金融市場充滿極高隨機性雜訊，模型 AUC 只要突破 {MLDefault.HIGH_AUC} 即屬華爾街頂尖水準)*
            """)

    st.markdown("---")

    st.markdown("### 📰 Gemini 新聞情緒")
    sentiment = result[APIKey.SENTIMENT.value]
    score = sentiment[OracleCol.SCORE.value]

    sentiment_color = Color.RED.value if score >= 7 else Color.GREEN.value if score <= 3 else Color.ORANGE.value
    st.info(f"**情緒分數：:{sentiment_color}[{score} / 10]** \n\n**判讀理由：** {sentiment[OracleCol.REASON.value]}")

    st.markdown("### 🤖 總裁財報")
    st.warning(result[APIKey.REPORT.value], icon="💡")