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


def render_batch_report(df_summary: pd.DataFrame, all_reports: list, name_map: dict):
    """
    渲染批次決策總表與各標的詳細摺疊財報
    """
    if df_summary.empty or not all_reports:
        st.info("目前尚無批次報告資料。")
        return

    display_df = df_summary.copy()
    if "標的" in display_df.columns:
        display_df["標的"] = display_df["標的"].apply(
            lambda x: f"{x} {name_map.get(x, '')}".strip()
        )

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

        ch_name = name_map.get(ticker, "")
        icon = "🛒" if action == TradeDecision.BUY.value else "💸" if action == TradeDecision.SELL.value else "🛡️"

        title = f"{icon} 【{ticker} {ch_name}】 AI 決策：{action}"

        with st.expander(title.strip()):
            render_report(rep['raw_result'])