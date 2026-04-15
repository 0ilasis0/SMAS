import streamlit as st

from bt.account import Account
from data.const import StockCol
from ui.const import SessionKey
from ui.stock_names import get_tw_stock_mapping

from .data import load_portfolio, save_portfolio
from .dialogs import (cash_operation_dialog, create_sub_portfolio_dialog,
                      fund_transfer_dialog, history_dialog, trade_dialog)


def render_portfolio_page(db_manager=None):
    st.title("💼 資產管理中心")

    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    if db_manager:
        for sp in account.sub_portfolios.values():
            for ticker, pos_obj in sp.positions.items():
                if pos_obj.shares > 0:
                    try:
                        df_latest = db_manager.get_daily_data(ticker)
                        if not df_latest.empty:
                            pos_obj.current_price = float(df_latest[StockCol.CLOSE.value].iloc[-1])
                    except Exception: pass

    # ==========================================
    # 頂部：總資產儀表板
    # ==========================================
    st.markdown("### 🌐 系統總覽")
    col1, col2, col3, col4 = st.columns([2, 2, 2.5, 1.5])
    col1.metric("👑 預估總資產", f"${account.total_equity:,.0f}")
    col2.metric("🎯 總持股成本", f"${account.total_cost_value:,.0f}")
    col3.metric("💧 未分配流動資金", f"${account.unallocated_cash:,.0f}", help="目前未被專屬組合包鎖定的系統閒置現金，可供共用組合包使用。")

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏦 總資金存取", use_container_width=True):
            cash_operation_dialog()

    st.markdown("---")

    c_head1, c_head2 = st.columns([4, 1])
    c_head1.subheader("📦 投資組合包明細")
    with c_head2:
        if st.button("➕ 新增組合包", type="primary", use_container_width=True):
            create_sub_portfolio_dialog()

    if not account.sub_portfolios:
        st.info("目前尚無任何組合包。點擊右上方「➕ 新增組合包」開始建立您的第一個投資組合！")
        return

    name_map = get_tw_stock_mapping()

    # ==========================================
    # 迴圈渲染每個組合包
    # ==========================================
    for sp_id, sp in account.sub_portfolios.items():
        sp_market_val = sp.total_market_value
        sp_cost_val = sp.total_cost_value
        sp_pnl = sp_market_val - sp_cost_val

        if sp_pnl >= 0:
            pnl_str = f":green[獲利: +${sp_pnl:,.0f}]"
            icon = "🟢"
        else:
            pnl_str = f":red[虧損: -${abs(sp_pnl):,.0f}]"
            icon = "🔴"

        expander_title = f"{icon} 組合包：{sp.name} | 市值: ${sp_market_val:,.0f} | {pnl_str}"

        with st.expander(expander_title, expanded=True):
            # --- 🌟 組合包頂部狀態列 (加入切換按鈕) ---
            sc1, sc2, sc3, sc4 = st.columns([2.5, 1, 1.5, 1.5])

            with sc1:
                if sp.use_shared_cash:
                    st.markdown("🚰 資金來源：**共用總資金**")
                else:
                    st.markdown(f"🔒 資金來源：**專屬資金** (餘額: **${sp.allocated_cash:,.0f}**)")

            with sc2:
                # 🌟 資金來源切換按鈕
                if sp.use_shared_cash:
                    if st.button("切換為專屬資金", key=f"tg_{sp_id}", help="將停止共用流動資金，改為獨立專屬資金 (初始餘額為 0)。", use_container_width=True):
                        sp.use_shared_cash = False
                        sp.allocated_cash = 0.0
                        save_portfolio(account)
                        st.rerun()
                else:
                    if st.button("切換為共用資金", key=f"tg_{sp_id}", help="將剩餘專屬資金退回，並改為共用未分配流動資金。", use_container_width=True):
                        sp.use_shared_cash = True
                        # 因為 use_shared_cash 變 true，它的 allocated_cash 在計算 unallocated_cash 時就不會被扣除，等同於自動退回大水庫！
                        sp.allocated_cash = 0.0
                        save_portfolio(account)
                        st.rerun()

            with sc3:
                # 只有專屬資金才顯示劃撥按鈕
                if not sp.use_shared_cash:
                    if st.button("🔄 劃撥資金", key=f"fund_{sp_id}", use_container_width=True):
                        fund_transfer_dialog(sp_id)
                else:
                    st.write("") # 佔位

            with sc4:
                if st.button("⚖️ 新增交易", key=f"trade_{sp_id}", type="primary", use_container_width=True):
                    trade_dialog(sp_id, db_manager)

            # --- 組合包庫存明細表 ---
            active_tickers = [t for t, p in sp.positions.items() if p.shares > 0 or len(p.history) > 0]

            if not active_tickers:
                st.info("此組合包目前無庫存。")
            else:
                hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([2, 1.5, 1.5, 2, 2, 2])
                hc1.caption("股票代號")
                hc2.caption("庫存 (股)")
                hc3.caption("均價")
                hc4.caption("市值")
                hc5.caption("未實現損益")
                hc6.caption("操作")
                st.divider()

                for ticker in sorted(active_tickers):
                    pos_obj = sp.get_position(ticker)
                    market_val = pos_obj.market_value
                    cost_val = pos_obj.cost_value
                    pnl = market_val - cost_val
                    pnl_pct = (pnl / cost_val) if cost_val > 0 else 0.0

                    pnl_color = "#00cc66" if pnl >= 0 else "#ff4b4b"
                    pnl_text = f"<span style='color: {pnl_color}; font-weight: bold;'>${pnl:,.0f} ({pnl_pct:.2%})</span>"
                    ch_name = name_map.get(ticker, "")

                    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns([2, 1.5, 1.5, 2, 2, 2])
                    rc1.markdown(f"**{ticker}** <br> <span style='font-size:0.85em; color:gray;'>{ch_name}</span>", unsafe_allow_html=True)
                    rc2.markdown(f"{pos_obj.shares:,}")
                    rc3.markdown(f"${pos_obj.avg_cost:,.2f}")
                    rc4.markdown(f"${market_val:,.0f}")
                    rc5.markdown(pnl_text, unsafe_allow_html=True)

                    with rc6:
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("交易", key=f"t_{sp_id}_{ticker}", use_container_width=True):
                                trade_dialog(sp_id, db_manager, prefill_ticker=ticker)
                        with btn_col2:
                            if st.button("歷史", key=f"h_{sp_id}_{ticker}", use_container_width=True):
                                history_dialog(sp_id, ticker)

                    st.markdown("<hr style='margin: 0; opacity: 0.2;'>", unsafe_allow_html=True)