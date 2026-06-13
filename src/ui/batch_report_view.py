import pandas as pd
import streamlit as st

from bt.const import TradeDecision
from ui.report import render_report


def highlight_action(row):
    """
    Pandas Styler 輔助函數：根據 AI 決策給予整列看盤軟體風的背景顏色提示
    """
    action = row.get("AI 決策", "")

    bg_color = ""
    if action == TradeDecision.BUY.value:
        bg_color = "background-color: rgba(255, 75, 75, 0.15)"  # 淺紅背景
    elif action == TradeDecision.SELL.value:
        bg_color = "background-color: rgba(33, 195, 84, 0.15)"  # 淺綠背景
    elif action == TradeDecision.HOLD.value:
        bg_color = "background-color: rgba(128, 128, 128, 0.1)" # 淺灰背景
    elif action == "尚未訓練":
        bg_color = "background-color: rgba(255, 165, 0, 0.15)"  # 警告橘色背景

    return [bg_color] * len(row)


def render_batch_report(df_summary: pd.DataFrame, all_reports: list):
    """
    渲染批次決策總表與各標的詳細摺疊財報
    """
    if df_summary.empty or not all_reports:
        st.info("目前尚無批次報告資料。")
        return

    st.markdown("### 🎯 今日作戰建議總表")

    # 1. 渲染頂部智慧排序與高亮的總結表格
    styled_df = df_summary.style.apply(highlight_action, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📖 各標的詳細 AI 財報")

    # 2. 渲染下方摺疊面板，並直接嵌入原有財報畫面
    for rep in all_reports:
        ticker = rep['ticker']
        action = rep['action']

        icon = "🛒" if action == TradeDecision.BUY.value else "💸" if action == TradeDecision.SELL.value else "🛡️"

        with st.expander(f"{icon} 【{ticker}】 AI 決策：{action}"):
            # 直接將單檔數據塞入原有的渲染引擎，畫面完全一致
            render_report(rep['raw_result'])