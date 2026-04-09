import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from const import Color
from data.const import StockCol
from ui.const import SessionKey


@st.cache_data(ttl=3600)
def get_cached_market_data(ticker: str):
    from data.manager import DataManager
    db = DataManager()
    return db.get_aligned_market_data(ticker, []).tail(720)


def render_chart():
    """渲染中央 K 線圖 (回歸拉桿模式、支援後端重算對焦、自適應 Y 軸)"""
    with st.expander("📉 K 線走勢圖 (拉桿探索模式)", expanded=True):
        try:
            ctrl = st.session_state.get(SessionKey.CTRL_LIVE.value)
            current_ticker = st.session_state.get(SessionKey.CURRENT_TICKER.value)

            if not ctrl or not current_ticker:
                return

            df_recent = get_cached_market_data(current_ticker)

            if not df_recent.empty:
                # 建立水平排列的按鈕，作為快速對焦功能
                window_map = {
                    "近 1 個月": 20,
                    "近 3 個月": 60,
                    "近半年": 120,
                    "近 1 年": 240,
                    "全部 (近 3 年)": len(df_recent)
                }

                selected_window = st.radio(
                    "🎯 視野快速對焦 (修正 Y 軸壓縮)：",
                    options=list(window_map.keys()),
                    index=3,
                    horizontal=True
                )

                lookback_days = window_map[selected_window]
                lookback_days = min(lookback_days, len(df_recent))

                visible_df = df_recent.iloc[-lookback_days:]
                initial_start_date = visible_df.index[0]
                end_date = visible_df.index[-1]

                fig = go.Figure()

                # 繪圖時使用完整 df_recent，這樣拉桿才能往回拉到 3 年前
                fig.add_trace(go.Candlestick(x=df_recent.index,
                                             open=df_recent[StockCol.OPEN.value],
                                             high=df_recent[StockCol.HIGH.value],
                                             low=df_recent[StockCol.LOW.value],
                                             close=df_recent[StockCol.CLOSE.value],
                                             increasing_line_color=Color.RED.value,
                                             decreasing_line_color=Color.GREEN.value,
                                             name='K線'))

                # 計算完整均線
                ma5 = df_recent[StockCol.CLOSE.value].rolling(window=5).mean().bfill()
                ma20 = df_recent[StockCol.CLOSE.value].rolling(window=20).mean().bfill()
                ma60 = df_recent[StockCol.CLOSE.value].rolling(window=60).mean().bfill()
                ma240 = df_recent[StockCol.CLOSE.value].rolling(window=240).mean().bfill()

                fig.add_trace(go.Scatter(x=df_recent.index, y=ma5, line=dict(color=Color.ORANGE.value, width=1.2), name='5MA'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma20, line=dict(color=Color.PURPLE.value, width=1.2), name='20MA'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma60, line=dict(color=Color.BLUE.value, width=1.2), name='60MA'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma240, line=dict(color=Color.WHITE.value, width=1.2), name='240MA'))

                # 假日斷點處理
                all_dates = pd.date_range(start=df_recent.index.min(), end=df_recent.index.max())
                missing_dates = all_dates.difference(df_recent.index)
                missing_dates_str = missing_dates.strftime('%Y-%m-%d').tolist()

                # Y 軸區間計算 (基於使用者選擇的對焦區間)
                local_max = visible_df[StockCol.HIGH.value].max()
                local_min = visible_df[StockCol.LOW.value].min()

                # 包含 240MA 以免在趨勢中斷頭
                visible_ma240 = ma240.iloc[-lookback_days:]
                if not visible_ma240.dropna().empty:
                    local_max = max(local_max, visible_ma240.max())
                    local_min = min(local_min, visible_ma240.min())

                amplitude = local_max - local_min
                padding = amplitude * 0.10 if amplitude != 0 else local_max * 0.05
                y_range = [local_min - padding, local_max + padding]

                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=550,
                    xaxis_rangeslider_visible=True,
                    xaxis_rangeslider_thickness=0.08,
                    legend=dict(orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.6)"),
                    xaxis=dict(
                        rangebreaks=[dict(values=missing_dates_str)],
                        range=[initial_start_date, end_date], # 初始視野鎖定在對焦區間
                        showgrid=True, gridcolor="rgba(200, 200, 200, 0.2)"
                    ),
                    yaxis=dict(
                        range=y_range, # Y 軸鎖定在對焦區間的最佳高度
                        fixedrange=False, # 允許手動微調 Y 軸
                        showgrid=True,
                        gridcolor="rgba(200, 200, 200, 0.2)",
                        zeroline=False
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'scrollZoom': True, 'displayModeBar': False, 'displaylogo': False}
                )
        except Exception as e:
            st.caption(f"無法渲染 K 線圖: {e}")
