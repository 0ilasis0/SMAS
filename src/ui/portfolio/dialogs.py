import time
from datetime import datetime

import pandas as pd
import streamlit as st

from bt.account import Account, SubPortfolio
from bt.const import TradeDecision
from bt.params import TaxRate
from data.const import StockCol
from ui.base import UIActionMapper, get_smart_tw_ticker, is_valid_ticker
from ui.const import HistoryKey, PortfolioCol, SessionKey, UIFormat
from ui.params import AccountLimit

from .data import (get_active_buys, load_portfolio, recalculate_position,
                   save_portfolio)


# ==========================================
# 1. 組合包專屬操作 Dialogs
# ==========================================
@st.dialog("📦 建立新組合包")
def create_sub_portfolio_dialog():
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    st.markdown("請設定新投資組合的名稱與資金來源。")

    sp_name = st.text_input("組合包名稱", placeholder="例如：AI 飆股策略、傳產避險池...")
    if not sp_name: return

    if sp_name in account.sub_portfolios:
        st.error("❌ 此名稱已存在，請換一個名稱。")
        return

    st.markdown("---")
    st.markdown("##### 💰 資金來源設定")

    unallocated = account.unallocated_cash

    fund_source = st.radio(
        "請選擇此組合包的扣款方式：",
        ["💧 共用總資金 (未分配流動資金)", "🔒 設定專屬資金"],
        help="共用總資金代表與其他組合包共享流動資金；專屬資金則會將現金獨立鎖定，避免被其他組合包買光。"
    )

    use_shared = fund_source.startswith("💧")
    allocated_amount = 0.0

    if not use_shared:
        if unallocated <= 0:
            st.warning("⚠️ 目前未分配流動資金為 $0，無法設定專屬資金！請先選擇「共用總資金」，日後再進行資金劃撥。")
            use_shared = True
        else:
            # 專屬資金提撥的快捷按鈕與 Session Key
            temp_key = "create_sp_alloc_amount"
            if temp_key not in st.session_state:
                st.session_state[temp_key] = 0.0

            st.caption("⚡ 快捷輸入")
            btn_cols = st.columns(4)
            if btn_cols[0].button("+ 10 萬", key="c1", use_container_width=True): st.session_state[temp_key] += 100_000.0
            if btn_cols[1].button("+ 100 萬", key="c2", use_container_width=True): st.session_state[temp_key] += 1_000_000.0
            if btn_cols[2].button("+ 1000 萬", key="c3", use_container_width=True): st.session_state[temp_key] += 10_000_000.0
            if btn_cols[3].button("歸零", key="c4", use_container_width=True): st.session_state[temp_key] = 0.0

            allocated_amount = st.number_input(
                f"請輸入提撥金額 (目前流動資金上限: ${unallocated:,.0f})",
                min_value=0.0,
                max_value=float(unallocated),
                step=10_000.0,
                key=temp_key
            )

            # 即時千分位與中文單位預覽
            amount_str = f"${allocated_amount:,.0f}"
            if allocated_amount >= 1_000_000_000:
                amount_str += f" ({allocated_amount / 1_000_000_000:.2f} 億)"
            elif allocated_amount >= 10_000:
                amount_str += f" ({allocated_amount / 10_000:.1f} 萬)"

            st.info(f"💡 目前設定金額： **{amount_str}**")

    if st.button("✅ 確認建立", type="primary", use_container_width=True):
        new_sp = SubPortfolio(
            name=sp_name,
            use_shared_cash=use_shared,
            allocated_cash=allocated_amount
        )
        account.sub_portfolios[sp_name] = new_sp

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        st.session_state["CURRENT_SUB_PORTFOLIO"] = sp_name

        # 執行完畢後清理 temp_key
        if not use_shared and "create_sp_alloc_amount" in st.session_state:
            del st.session_state["create_sp_alloc_amount"]

        st.toast(f"✅ 成功建立組合包：{sp_name}", icon="📦")
        time.sleep(0.5)
        st.rerun()

@st.dialog("🔄 資金劃撥")
def fund_transfer_dialog(sp_id: str):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    sp = account.get_sub_portfolio(sp_id)

    if sp.use_shared_cash:
        st.info("此組合包目前設定為「共用總資金」，無需手動劃撥。")
        return

    st.markdown(f"調整 **{sp.name}** 的專屬資金。")

    unallocated = account.unallocated_cash
    current_allocated = sp.allocated_cash

    st.write(f"💧 未分配流動資金：**${unallocated:,.0f}**")
    st.write(f"🔒 此組合目前專屬資金：**${current_allocated:,.0f}**")

    st.markdown("---")
    action = st.radio("操作類型", ["📥 從流動資金提撥進來", "📤 將資金退回流動資金"], horizontal=True)

    max_transfer = unallocated if action.startswith("📥") else current_allocated

    temp_key = f"fund_transfer_{sp_id}"
    if temp_key not in st.session_state:
        st.session_state[temp_key] = 0.0

    # 快捷金額按鈕
    st.caption("⚡ 快捷輸入")
    btn_cols = st.columns(4)
    if btn_cols[0].button("+ 10 萬", key="f1", use_container_width=True): st.session_state[temp_key] += 100_000.0
    if btn_cols[1].button("+ 100 萬", key="f2", use_container_width=True): st.session_state[temp_key] += 1_000_000.0
    if btn_cols[2].button("+ 1000 萬", key="f3", use_container_width=True): st.session_state[temp_key] += 10_000_000.0
    if btn_cols[3].button("歸零", key="f4", use_container_width=True): st.session_state[temp_key] = 0.0

    amount = st.number_input("金額 (NTD)", min_value=0.0, max_value=float(max_transfer), step=10_000.0, key=temp_key)

    # 即時千分位與中文單位預覽
    amount_str = f"${amount:,.0f}"
    if amount >= 1_000_000_000:
        amount_str += f" ({amount / 1_000_000_000:.2f} 億)"
    elif amount >= 10_000:
        amount_str += f" ({amount / 10_000:.1f} 萬)"

    st.info(f"💡 目前設定金額： **{amount_str}**")

    if st.button("確認劃撥", type="primary", use_container_width=True):
        if amount <= 0: return

        if action.startswith("📥"):
            if amount > unallocated:
                st.error(f"❌ 餘額不足！(目前未分配流動資金僅有 ${unallocated:,.0f})")
                return
            sp.allocated_cash += amount
            st.toast(f"✅ 已提撥 ${amount:,.0f} 至專屬資金。", icon="📥")
        else:
            if amount > current_allocated:
                st.error("❌ 餘額不足！")
                return
            sp.allocated_cash -= amount
            st.toast(f"✅ 已將 ${amount:,.0f} 退回未分配流動資金。", icon="📤")

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        del st.session_state[temp_key]
        time.sleep(0.5)
        st.rerun()

# ==========================================
# 2. 系統總現金存取 Dialog
# ==========================================
@st.dialog("🏦 系統總帳戶存取")
def cash_operation_dialog():
    st.markdown("這將影響整個系統的「總資金」。")
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    op_type = st.radio("操作類型", ["📥 存入總資金 (入金)", "📤 提出總資金 (出金)"], horizontal=True)

    temp_key = PortfolioCol.TEMP_CASH_AMOUNT.value
    if temp_key not in st.session_state:
        st.session_state[temp_key] = float(AccountLimit.MIN_MONEY)

    btn_cols = st.columns(4)
    if btn_cols[0].button("+ 10 萬", use_container_width=True): st.session_state[temp_key] += 100_000.0
    if btn_cols[1].button("+ 100 萬", use_container_width=True): st.session_state[temp_key] += 1_000_000.0
    if btn_cols[2].button("+ 1000 萬", use_container_width=True): st.session_state[temp_key] += 10_000_000.0
    if btn_cols[3].button("歸零", use_container_width=True): st.session_state[temp_key] = 0.0

    amount = st.number_input("金額 (NTD)", min_value=0.0, max_value=float(AccountLimit.MAX_MONEY), step=10_000.0, key=temp_key)

    amount_str = f"${amount:,.0f}"
    if amount >= 1_000_000_000:
        amount_str += f" ({amount / 1_000_000_000:.2f} 億)"
    elif amount >= 10_000:
        amount_str += f" ({amount / 10_000:.1f} 萬)"

    st.info(f"💡 目前設定金額： **{amount_str}**")
    st.markdown("---")

    if st.button("確認執行", type="primary", use_container_width=True):
        if amount <= 0: return

        if op_type.startswith("📤"):
            if amount > account.unallocated_cash:
                st.error(f"❌ 未分配流動資金不足！(目前可出金流動資金僅有 ${account.unallocated_cash:,.0f})")
                return
            account.total_cash -= amount
            st.success(f"✅ 成功提出 {amount:,.0f} 元。")
        else:
            account.total_cash += amount
            st.success(f"✅ 成功存入 {amount:,.0f} 元。")

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        del st.session_state[temp_key]
        time.sleep(0.5)
        st.rerun()

# ==========================================
# 3. 新增交易 Dialog (扣款提示更新)
# ==========================================
@st.dialog("⚖️ 新增交易")
def trade_dialog(sp_id: str, db_manager, prefill_ticker: str = "", prefill_action: str = None, prefill_price: float = None, prefill_shares: int = None):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    sp = account.get_sub_portfolio(sp_id)

    st.caption(f"📂 目前操作組合包：**{sp.name}**")

    raw_ticker = st.text_input("🔍 股票代號 (輸入後按 Enter 抓取現價)", value=prefill_ticker, placeholder="例如: 2330 或 3105.TWO")
    if not raw_ticker: return

    # 使用智慧解析器找出真正的代號
    ticker = get_smart_tw_ticker(raw_ticker)
    if not ticker:
        st.error(f"❌ 找不到標的 {raw_ticker}！(無論上市或上櫃皆無此代號)")
        return

    pos_obj = sp.get_position(ticker)

    fetched_price = pos_obj.avg_cost if pos_obj.avg_cost > 0 else 10.0
    if prefill_price is not None and prefill_price > 0: fetched_price = float(prefill_price)
    elif db_manager:
        try:
            df = db_manager.get_daily_data(ticker)
            if not df.empty:
                fetched_price = float(df[StockCol.CLOSE.value].iloc[-1])
                pos_obj.current_price = fetched_price
        except Exception: pass

    st.markdown("---")
    active_buys = get_active_buys(pos_obj.history)
    if active_buys and pos_obj.shares > 0:
        st.caption(f"💡 目前持股均價 **${pos_obj.avg_cost:.2f}** 的構成批次：")
        df_buys = pd.DataFrame(active_buys)[[HistoryKey.DATE.value, HistoryKey.PRICE.value, HistoryKey.SHARES.value, HistoryKey.TOTAL.value]]
        df_buys.columns = ["買進時間", "買進單價", "股數", "含費總成本"]
        st.dataframe(df_buys, use_container_width=True, hide_index=True)
    elif pos_obj.shares <= 0:
        st.info("此組合包目前無該檔股票庫存。")

    st.markdown("---")
    action = st.radio("交易動作", UIActionMapper.get_options(), index=1 if prefill_action and str(prefill_action).lower() == TradeDecision.SELL.value else 0, horizontal=True)
    is_buy = UIActionMapper.is_buy(action)

    col1, col2 = st.columns(2)
    with col1: trade_price = st.number_input("成交單價 (元)", min_value=0.01, value=fetched_price, step=0.5, format="%.2f")
    with col2:
        default_shares = prefill_shares if prefill_shares is not None else 1000
        if not is_buy and default_shares > pos_obj.shares and pos_obj.shares > 0: default_shares = pos_obj.shares
        trade_shares = st.number_input("成交股數 (股)", min_value=1, value=int(max(1, default_shares)), step=1000)

    base_amount = trade_price * trade_shares
    fee = int(max(TaxRate.MIN_FEE, base_amount * TaxRate.FEE_RATE))
    tax = int(base_amount * TaxRate.TAX_RATE) if not is_buy else 0
    total_settlement = base_amount + fee if is_buy else base_amount - fee - tax

    st.info(f"💵 預估{'應付' if is_buy else '應收'}交割：**${total_settlement:,.0f}**")

    msg_placeholder = st.empty()
    if st.button("確認送出交易", type="primary", use_container_width=True):
        if pos_obj.shares == 0 and len(pos_obj.history) == 0:
            if not is_valid_ticker(ticker):
                msg_placeholder.error(f"❌ 找不到標的 {ticker}！")
                return

        if is_buy:
            if sp.use_shared_cash:
                if total_settlement > account.unallocated_cash:
                    msg_placeholder.error(f"❌ 未分配流動資金不足！(餘額: ${account.unallocated_cash:,.0f})")
                    return
                account.total_cash -= total_settlement
            else:
                if total_settlement > sp.allocated_cash:
                    msg_placeholder.error(f"❌ 專屬資金不足！(餘額: ${sp.allocated_cash:,.0f})")
                    return
                sp.allocated_cash -= total_settlement
                account.total_cash -= total_settlement

            old_total_cost = pos_obj.shares * pos_obj.avg_cost
            pos_obj.shares += trade_shares
            pos_obj.avg_cost = (old_total_cost + total_settlement) / pos_obj.shares
            if ticker not in sp.watch_tickers: sp.watch_tickers.append(ticker)
        else:
            if trade_shares > pos_obj.shares:
                msg_placeholder.error(f"❌ 庫存不足！(僅持有: {pos_obj.shares} 股)")
                return
            pos_obj.shares -= trade_shares
            pos_obj.avg_cost = pos_obj.avg_cost if pos_obj.shares > 0 else 0.0

            if sp.use_shared_cash: account.total_cash += total_settlement
            else: sp.allocated_cash += total_settlement; account.total_cash += total_settlement

        trade_record = {
            HistoryKey.DATE.value: datetime.now().strftime(UIFormat.DATETIME_FORMAT.value),
            HistoryKey.ACTION.value: TradeDecision.BUY.value if is_buy else TradeDecision.SELL.value,
            HistoryKey.PRICE.value: trade_price,
            HistoryKey.SHARES.value: trade_shares,
            HistoryKey.FEE.value: fee,
            HistoryKey.TAX.value: tax,
            HistoryKey.TOTAL.value: total_settlement
        }
        pos_obj.history.append(trade_record)

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        st.rerun()

# ==========================================
# 4. 歷史資料 Dialog (扣款提示更新)
# ==========================================
@st.dialog("📜 歷史資料與帳務檢視", width="large")
def history_dialog(sp_id: str, ticker: str):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    sp = account.get_sub_portfolio(sp_id)
    pos_obj = sp.get_position(ticker)

    st.markdown(f"檢視 **【{sp.name}】** 中 **{ticker}** 的歷史明細。")

    is_empty_history = len(pos_obj.history) == 0
    if is_empty_history: st.info("尚無任何歷史紀錄。")
    else:
        df = pd.DataFrame(pos_obj.history)
        for col in [HistoryKey.FEE.value, HistoryKey.TAX.value]:
            if col not in df.columns: df[col] = 0
        if HistoryKey.ACTION.value in df.columns:
            df[HistoryKey.ACTION.value] = df[HistoryKey.ACTION.value].astype(str).str.lower().map(UIActionMapper.get_map()).fillna(df[HistoryKey.ACTION.value])

        df_display = df[[HistoryKey.DATE.value, HistoryKey.ACTION.value, HistoryKey.PRICE.value, HistoryKey.SHARES.value, HistoryKey.FEE.value, HistoryKey.TAX.value, HistoryKey.TOTAL.value]].copy()
        df_display.columns = ["時間", "動作", "單價", "股數", "手續費", "交易稅", "交割淨額"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if is_empty_history: st.button("↩️ 撤銷最新一筆", disabled=True, use_container_width=True)
        else:
            if st.button("↩️ 撤銷最新一筆交易", use_container_width=True):
                last_trade = pos_obj.history[-1]
                action = str(last_trade.get(HistoryKey.ACTION.value)).lower()
                total_settlement = float(last_trade.get(HistoryKey.TOTAL.value, 0.0))

                if action == TradeDecision.SELL.value:
                    if sp.use_shared_cash and account.unallocated_cash < total_settlement:
                        st.error(f"❌ 撤銷失敗！未分配流動資金不足 (${total_settlement:,.0f})。")
                        return
                    elif not sp.use_shared_cash and sp.allocated_cash < total_settlement:
                        st.error(f"❌ 撤銷失敗！專屬資金不足 (${total_settlement:,.0f})。")
                        return

                pos_obj.history.pop()

                if action == TradeDecision.BUY.value:
                    if sp.use_shared_cash: account.total_cash += total_settlement
                    else: sp.allocated_cash += total_settlement; account.total_cash += total_settlement
                else:
                    if sp.use_shared_cash: account.total_cash -= total_settlement
                    else: sp.allocated_cash -= total_settlement; account.total_cash -= total_settlement

                new_shares, new_avg_cost = recalculate_position(pos_obj.history)
                pos_obj.shares = new_shares
                pos_obj.avg_cost = new_avg_cost

                if new_shares == 0 and len(pos_obj.history) == 0 and ticker in sp.positions:
                    del sp.positions[ticker]

                save_portfolio(account)
                st.session_state[SessionKey.PORTFOLIO.value] = account
                st.rerun()

    with col2:
        if pos_obj.shares > 0: st.button("🗑️ 徹底隱藏並刪除", disabled=True, use_container_width=True)
        else:
            if st.button("🗑️ 徹底隱藏並刪除此標的", type="primary", use_container_width=True):
                if ticker in sp.positions: del sp.positions[ticker]
                save_portfolio(account)
                st.session_state[SessionKey.PORTFOLIO.value] = account
                st.rerun()

# ==========================================
# 5. 組合包專屬設定 Dialog
# ==========================================
@st.dialog("⚙️ 組合包設定")
def sub_portfolio_settings_dialog(sp_id: str):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    if sp_id not in account.sub_portfolios:
        st.error("找不到該組合包。")
        return

    sp = account.get_sub_portfolio(sp_id)

    st.markdown("#### ✏️ 重新命名")
    new_name = st.text_input("輸入新名稱", value=sp.name, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### 💰 資金來源切換")
    current_source_index = 0 if sp.use_shared_cash else 1
    fund_source = st.radio(
        "扣款方式：",
        ["💧 共用總資金 (未分配流動資金)", "🔒 設定專屬資金"],
        index=current_source_index,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # 儲存變更按鈕
    if st.button("💾 儲存變更", type="primary", use_container_width=True):
        # 1. 處理重新命名
        if new_name != sp_id:
            if new_name in account.sub_portfolios:
                st.error("❌ 名稱已存在，請換一個名稱！")
                return
            sp.name = new_name
            # 更換字典的 Key
            account.sub_portfolios[new_name] = account.sub_portfolios.pop(sp_id)
            # 如果目前正在操作這個組合包，同步更新 Session
            if st.session_state.get("CURRENT_SUB_PORTFOLIO") == sp_id:
                st.session_state["CURRENT_SUB_PORTFOLIO"] = new_name

        # 2. 處理資金切換
        new_use_shared = fund_source.startswith("💧")
        if new_use_shared != sp.use_shared_cash:
            sp.use_shared_cash = new_use_shared
            sp.allocated_cash = 0.0 # 切換時，專屬資金重置為 0 (若退回共用，資金會自動回到大水庫)

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        st.toast("✅ 設定已更新！", icon="💾")
        time.sleep(0.5)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 危險操作區
    with st.expander("⚠️ 危險操作：刪除組合包"):
        st.warning(f"刪除組合包後，其內部所有的庫存將被強制以**最新收盤價**結算，並將剩餘資金全數退回系統的「未分配流動資金」。\n此操作**無法復原**！")
        confirm_delete = st.checkbox("我了解風險，確認刪除")

        if st.button("🚨 確認刪除並結算", disabled=not confirm_delete, use_container_width=True):
            liquidation_value = sp.total_market_value
            returned_cash = sp.allocated_cash + liquidation_value

            # 將退回的資金加到系統總資金
            account.total_cash += returned_cash
            del account.sub_portfolios[sp_id]

            save_portfolio(account)
            st.session_state[SessionKey.PORTFOLIO.value] = account

            # 如果刪除的是當前選中的組合包，清空焦點
            if st.session_state.get("CURRENT_SUB_PORTFOLIO") == sp_id:
                st.session_state["CURRENT_SUB_PORTFOLIO"] = None
                st.session_state[SessionKey.CURRENT_TICKER.value] = None
                st.session_state[SessionKey.CTRL_LIVE.value] = None

            st.toast(f"✅ 已刪除組合包並退回結算資金 ${returned_cash:,.0f}", icon="🗑️")
            time.sleep(1.0)
            st.rerun()