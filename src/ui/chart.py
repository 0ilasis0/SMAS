import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from const import Color
from data.const import StockCol


@st.cache_data(ttl=3600)
def get_cached_market_data(ticker: str):
    from data.manager import DataManager
    db = DataManager()
    return db.get_aligned_market_data(ticker, []).tail(720)

def render_chart():
    """渲染中央 K 線圖 (支援動態縮放、全中文月份、無縫接合斷點)"""
    with st.expander("📉 近期走勢圖 (預設顯示近 1 年，可自由縮放檢視歷史)", expanded=True):
        try:
            ctrl = st.session_state.ctrl_live
            if not ctrl: return

            # 使用快取函數拿取資料，避免每次切換 Tab 都重查 DB
            df_recent = get_cached_market_data(st.session_state.current_ticker)

            if not df_recent.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_recent.index,
                                             open=df_recent[StockCol.OPEN],
                                             high=df_recent[StockCol.HIGH],
                                             low=df_recent[StockCol.LOW],
                                             close=df_recent[StockCol.CLOSE],
                                             increasing_line_color=Color.RED,
                                             decreasing_line_color=Color.GREEN,
                                             name='K線'))

                ma5 = df_recent[StockCol.CLOSE].rolling(window=5).mean().bfill()
                ma20 = df_recent[StockCol.CLOSE].rolling(window=20).mean().bfill()
                ma60 = df_recent[StockCol.CLOSE].rolling(window=60).mean().bfill()
                ma240 = df_recent[StockCol.CLOSE].rolling(window=240).mean().bfill()

                fig.add_trace(go.Scatter(x=df_recent.index, y=ma5, line=dict(color=Color.ORANGE, width=1.5), name='5MA(週線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma20, line=dict(color=Color.PURPLE, width=1.5), name='20MA(月線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma60, line=dict(color=Color.BLUE, width=1.5), name='60MA(季線)'))
                fig.add_trace(go.Scatter(x=df_recent.index, y=ma240, line=dict(color=Color.WHITE, width=1.5), name='240MA(年線)'))

                all_dates = pd.date_range(start=df_recent.index.min(), end=df_recent.index.max())
                missing_dates = all_dates.difference(df_recent.index)
                missing_dates_str = missing_dates.strftime('%Y-%m-%d').tolist()

                first_days_of_month = df_recent.groupby([df_recent.index.year, df_recent.index.month]).head(1)
                tickvals = first_days_of_month.index
                ticktext = [f"{d.year}年{d.month}月" if d.month == 1 else f"{d.month}月" for d in tickvals]

                initial_start_date = df_recent.index[-240] if len(df_recent) >= 240 else df_recent.index[0]
                end_date = df_recent.index[-1]

                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=450,
                    xaxis_rangeslider_visible=True,
                    legend=dict(orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.6)"),
                    xaxis=dict(
                        rangebreaks=[dict(values=missing_dates_str)],
                        range=[initial_start_date, end_date],
                        tickmode='array', tickvals=tickvals, ticktext=ticktext,
                        showgrid=True, gridcolor="rgba(200, 200, 200, 0.2)"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'scrollZoom': True, 'displayModeBar': False, 'displaylogo': False}
                )
        except Exception as e:
            st.caption(f"無法渲染 K 線圖: {e}")