import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from bt.account import Account, Position
from bt.const import TradeDecision
from bt.params import TaxRate
from data.const import StockCol
from path import PathConfig
from ui.base import UIActionMapper, is_valid_ticker
from ui.const import (EncodingConst, HistoryKey, PortfolioCol, SessionKey,
                      UIFormat)
from ui.params import AccountLimit
from ui.stock_names import get_tw_stock_mapping


# ==========================================
# 1. 資料層：直接回傳/儲存 Account 物件
# ==========================================
def load_portfolio() -> Account:
    """從 JSON 讀取資料，並直接實例化為 Account 物件"""
    account = Account()
    if os.path.exists(PathConfig.PORTFOLIO):
        try:
            with open(PathConfig.PORTFOLIO, "r", encoding=EncodingConst.UTF8.value) as f:
                data = json.load(f)
                account.cash = data.get(PortfolioCol.GLOBAL_CASH.value, 0.0)

                # 將字典轉換為 Position 物件
                positions_data = data.get(PortfolioCol.POSITIONS.value, {})
                for ticker, pos_dict in positions_data.items():
                    account.positions[ticker] = Position(
                        shares=pos_dict.get(PortfolioCol.SHARES.value, 0),
                        avg_cost=pos_dict.get(PortfolioCol.AVG_COST.value, 0.0),
                        current_price=pos_dict.get(PortfolioCol.AVG_COST.value, 0.0), # 預設先用成本價當現價
                        history=pos_dict.get(PortfolioCol.HISTORY.value, [])
                    )
        except Exception as e:
            st.error(f"讀取資金檔發生錯誤，將使用空帳戶: {e}")

    return account

def save_portfolio(account: Account):
    """將 Account 物件序列化回 JSON 儲存"""
    try:
        os.makedirs(os.path.dirname(PathConfig.PORTFOLIO), exist_ok=True)

        # 將物件轉回字典結構
        data_to_save = {
            PortfolioCol.GLOBAL_CASH.value: account.cash,
            PortfolioCol.POSITIONS.value: {
                ticker: {
                    PortfolioCol.SHARES.value: pos.shares,
                    PortfolioCol.AVG_COST.value: pos.avg_cost,
                    PortfolioCol.HISTORY.value: pos.history
                }
                for ticker, pos in account.positions.items()
            }
        }

        with open(PathConfig.PORTFOLIO, "w", encoding=EncodingConst.UTF8.value) as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"❌ 資金檔存檔失敗: {e}")

def recalculate_position(history: list) -> tuple[int, float]:
    """
    動態帳務重算引擎 (嚴格平均成本法)
    """
    current_shares = 0
    # 在會計上，我們追蹤「總投入本金(不含已實現損益)」，以便計算平均成本
    current_total_cost = 0.0

    STANDARD_BUY = TradeDecision.BUY.value
    STANDARD_SELL = TradeDecision.SELL.value

    for r in history:
        action = str(r.get(HistoryKey.ACTION.value, STANDARD_BUY)).lower()
        shares = int(r.get(HistoryKey.SHARES.value, 0))
        total_settlement = float(r.get(HistoryKey.TOTAL.value, 0.0))

        if action == STANDARD_BUY:
            current_shares += shares
            current_total_cost += total_settlement

        elif action == STANDARD_SELL:
            if current_shares > 0:
                # 賣出時，先算出當前的平均成本
                avg_cost = current_total_cost / current_shares

                # 扣除股數
                current_shares -= shares

                if current_shares <= 0:
                    current_shares = 0
                    current_total_cost = 0.0
                else:
                    current_total_cost = current_shares * avg_cost

    # 最終防呆計算
    final_avg_cost = current_total_cost / current_shares if current_shares > 0 else 0.0
    return current_shares, final_avg_cost

def get_active_buys(history: list) -> list:
    """找出構成「當前庫存均價」的所有有效買進批次"""
    temp_shares = 0
    last_zero_idx = -1
    STANDARD_BUY = TradeDecision.BUY.value

    for i, r in enumerate(history):
        if str(r.get(HistoryKey.ACTION.value)).lower() == STANDARD_BUY:
            temp_shares += int(r.get(HistoryKey.SHARES.value, 0))
        else:
            temp_shares -= int(r.get(HistoryKey.SHARES.value, 0))
            if temp_shares <= 0:
                temp_shares = 0
                last_zero_idx = i

    active_buys = []
    for i in range(last_zero_idx + 1, len(history)):
        r = history[i]
        if str(r.get(HistoryKey.ACTION.value)).lower() == STANDARD_BUY:
            active_buys.append(r)

    return active_buys

# ==========================================
# 2. UI 層：歷史資料編輯器
# ==========================================
@st.dialog("📜 歷史資料與帳務編修", width="large")
def history_dialog(ticker: str):
    st.markdown(f"管理 **{ticker}** 的歷史交易紀錄。您可以直接在表格中點選該列並按 `Delete` 來刪除錯誤的紀錄。")
    st.warning("⚠️ 刪除歷史紀錄將會**自動重新計算**您的庫存數量與平均成本，但**不會**退還/扣除可用現金。")

    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    pos_obj = account.get_position(ticker)

    if not pos_obj.history:
        st.info("尚無任何歷史紀錄。")
        return

    df = pd.DataFrame(pos_obj.history)
    for col in [HistoryKey.FEE.value, HistoryKey.TAX.value]:
        if col not in df.columns:
            df[col] = 0

    if HistoryKey.ACTION.value in df.columns:
        df[HistoryKey.ACTION.value] = df[HistoryKey.ACTION.value].astype(str).str.lower().map(UIActionMapper.get_map()).fillna(df[HistoryKey.ACTION.value])

    df_display = df[[HistoryKey.DATE.value, HistoryKey.ACTION.value, HistoryKey.PRICE.value,
                     HistoryKey.SHARES.value, HistoryKey.FEE.value, HistoryKey.TAX.value, HistoryKey.TOTAL.value]].copy()
    df_display.columns = ["時間", "動作", "單價", "股數", "手續費", "交易稅", "交割淨額"]

    edited_df = st.data_editor(
        df_display,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "動作": st.column_config.SelectboxColumn(
                "動作", options=UIActionMapper.get_options(), required=True
            )
        }
    )

    if st.button("💾 儲存歷史變更並重新結算庫存", type="primary", use_container_width=True):
        new_history = []
        for _, row in edited_df.iterrows():
            new_history.append({
                HistoryKey.DATE.value: str(row["時間"]),
                HistoryKey.ACTION.value: UIActionMapper.to_core(str(row["動作"])),
                HistoryKey.PRICE.value: float(row["單價"]),
                HistoryKey.SHARES.value: int(row["股數"]),
                HistoryKey.FEE.value: int(row["手續費"]),
                HistoryKey.TAX.value: int(row["交易稅"]),
                HistoryKey.TOTAL.value: float(row["交割淨額"])
            })

        new_shares, new_avg_cost = recalculate_position(new_history)

        pos_obj.history = new_history
        pos_obj.shares = new_shares
        pos_obj.avg_cost = new_avg_cost

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account
        st.success(f"✅ 重算完成！目前庫存: {new_shares} 股 / 均價: {new_avg_cost:.2f} 元")
        st.rerun()

# ==========================================
# 3. 交易引擎：彈出式手動買賣視窗
# ==========================================
@st.dialog("⚖️ 新增交易", width="large")
def trade_dialog(db_manager, prefill_ticker: str = "", prefill_action: str = None, prefill_price: float = None, prefill_shares: int = None):
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    raw_ticker = st.text_input("🔍 股票代號 (輸入後按 Enter 抓取現價)", value=prefill_ticker, placeholder="例如: 2330.TW")
    if not raw_ticker: return

    ticker = raw_ticker.strip().upper()
    if not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        ticker += ".TW"

    pos_obj = account.get_position(ticker)

    fetched_price = pos_obj.avg_cost if pos_obj.avg_cost > 0 else 10.0
    if prefill_price is not None and prefill_price > 0:
        fetched_price = float(prefill_price)
    elif db_manager:
        try:
            df = db_manager.get_daily_data(ticker)
            if not df.empty:
                fetched_price = float(df[StockCol.CLOSE.value].iloc[-1])
                pos_obj.current_price = fetched_price # 順便更新最新價
        except Exception:
            pass

    st.markdown("---")
    active_buys = get_active_buys(pos_obj.history)
    if active_buys and pos_obj.shares > 0:
        st.caption(f"💡 目前持股均價 **${pos_obj.avg_cost:.2f}** 的構成批次 (有效買進明細)：")
        df_buys = pd.DataFrame(active_buys)
        df_buys = df_buys[[HistoryKey.DATE.value, HistoryKey.PRICE.value, HistoryKey.SHARES.value, HistoryKey.TOTAL.value]]
        df_buys.columns = ["買進時間", "買進單價", "股數", "含費總成本"]
        st.dataframe(df_buys, use_container_width=True, hide_index=True)
    elif pos_obj.shares <= 0:
        st.info("目前無庫存。")

    st.markdown("---")

    default_action_idx = 0
    if prefill_action and str(prefill_action).lower() == TradeDecision.SELL.value:
        default_action_idx = 1

    action = st.radio("交易動作", UIActionMapper.get_options(), index=default_action_idx, horizontal=True)
    is_buy = UIActionMapper.is_buy(action)

    col1, col2 = st.columns(2)
    with col1:
        trade_price = st.number_input("成交單價 (元)", min_value=0.01, value=fetched_price, step=0.5, format="%.2f")
    with col2:
        default_shares = prefill_shares if prefill_shares is not None else 1000
        if default_shares <= 0: default_shares = 1000
        if not is_buy and default_shares > pos_obj.shares and pos_obj.shares > 0:
            default_shares = pos_obj.shares

        trade_shares = st.number_input("成交股數 (股)", min_value=1, value=int(default_shares), step=1000)

    base_amount = trade_price * trade_shares
    fee = int(max(TaxRate.MIN_FEE, base_amount * TaxRate.FEE_RATE))
    tax = int(base_amount * TaxRate.TAX_RATE) if not is_buy else 0

    if is_buy:
        total_settlement = base_amount + fee
        st.info(f"💵 預估應付交割：**${total_settlement:,.0f}** (含手續費 ${fee})")
    else:
        total_settlement = base_amount - fee - tax
        st.info(f"💵 預估應收交割：**${total_settlement:,.0f}** (扣除手續費 ${fee}、交易稅 ${tax})")

    msg_placeholder = st.empty()
    if st.button("確認送出交易", type="primary", use_container_width=True):
        if pos_obj.shares == 0 and len(pos_obj.history) == 0:
            with st.spinner(f"正在驗證 {ticker} 是否存在..."):
                if not is_valid_ticker(ticker):
                    msg_placeholder.error(f"❌ 找不到標的 {ticker}！請確認代號是否正確。")
                    return

        if is_buy:
            if total_settlement > account.cash:
                msg_placeholder.error(f"❌ 可用現金不足！(應付: ${total_settlement:,.0f} / 餘額: ${account.cash:,.0f})")
                return
            old_total_cost = pos_obj.shares * pos_obj.avg_cost
            new_total_cost = old_total_cost + total_settlement
            pos_obj.shares += trade_shares
            pos_obj.avg_cost = new_total_cost / pos_obj.shares
            account.cash -= total_settlement
        else:
            if trade_shares > pos_obj.shares:
                msg_placeholder.error(f"❌ 庫存餘額不足！(目前僅持有: {pos_obj.shares} 股)")
                return
            pos_obj.shares -= trade_shares
            pos_obj.avg_cost = pos_obj.avg_cost if pos_obj.shares > 0 else 0.0
            account.cash += total_settlement

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
# 存取資金 Dialog
# ==========================================
@st.dialog("🏦 存取資金")
def cash_operation_dialog():
    st.markdown("請輸入您要存入或提出的金額。")
    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    op_type = st.radio("操作類型", ["📥 存入資金 (Deposit)", "📤 提出資金 (Withdraw)"], horizontal=True)

    temp_key = PortfolioCol.TEMP_CASH_AMOUNT.value
    if temp_key not in st.session_state:
        st.session_state[temp_key] = float(AccountLimit.MIN_MONEY)

    # 快捷金額按鈕
    st.caption("⚡ 快捷輸入")
    btn_cols = st.columns(4)
    if btn_cols[0].button("+ 10 萬", use_container_width=True):
        st.session_state[temp_key] += 100_000.0
    if btn_cols[1].button("+ 100 萬", use_container_width=True):
        st.session_state[temp_key] += 1_000_000.0
    if btn_cols[2].button("+ 1000 萬", use_container_width=True):
        st.session_state[temp_key] += 10_000_000.0
    if btn_cols[3].button("歸零", use_container_width=True):
        st.session_state[temp_key] = 0.0

    # 實際的輸入框
    amount = st.number_input(
        "金額 (NTD)",
        min_value=0.0,
        max_value=float(AccountLimit.MAX_MONEY),
        step=10_000.0,
        format="%.0f",
        key=temp_key
    )

    # 即時千分位預覽
    amount_str = f"${amount:,.0f}"
    if amount >= 1_000_000_000:
        amount_str += f" ({amount / 1_000_000_000:.2f} 億)"
    elif amount >= 10_000:
        amount_str += f" ({amount / 10_000:.0f} 萬)"

    st.info(f"💡 目前設定金額： **{amount_str}**")

    st.markdown("---")

    if st.button("確認執行", type="primary", use_container_width=True):
        if amount <= 0:
            st.warning("⚠️ 請輸入大於 0 的金額。")
            return

        if op_type.startswith("📤"):
            if amount > account.cash:
                st.error(f"❌ 餘額不足！(目前帳戶餘額僅有 ${account.cash:,.0f})")
                return
            account.cash -= amount
            st.success(f"✅ 成功提出 {amount:,.0f} 元。")
        else:
            account.cash += amount
            st.success(f"✅ 成功存入 {amount:,.0f} 元。")

        save_portfolio(account)
        st.session_state[SessionKey.PORTFOLIO.value] = account

        # 執行完畢後清理 temp 變數，避免下次點開 Dialog 時殘留
        del st.session_state[temp_key]
        time.sleep(0.5)
        st.rerun()

# ==========================================
# 4. 視圖層：資產管理中心主畫面
# ==========================================
def render_portfolio_page(db_manager=None):
    st.title("💼 資產管理中心")
    st.markdown("---")

    account: Account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())

    # 若有資料庫，幫所有現有庫存更新最新報價
    if db_manager:
        for ticker, pos_obj in account.positions.items():
            if pos_obj.shares > 0:
                try:
                    df_latest = db_manager.get_daily_data(ticker)
                    if not df_latest.empty:
                        latest_price = float(df_latest[StockCol.CLOSE.value].iloc[-1])
                        account.update_price(ticker, latest_price)
                except Exception:
                    pass

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
    col1.metric("👑 預估總資產", f"${account.total_equity:,.0f}")
    col2.metric("🎯 總成本", f"${account.total_cost_value:,.0f}")
    col3.metric("💵 可用現金", f"${account.cash:,.0f}")

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🏦 存取款", use_container_width=True):
                cash_operation_dialog()
        with c_btn2:
            if st.button("⚖️ 新增交易", type="primary", use_container_width=True):
                trade_dialog(db_manager, prefill_ticker="")

    st.markdown("---")
    st.subheader("📦 目前庫存明細")

    # 過濾出有持股的標的
    active_tickers = [t for t, p in account.positions.items() if p.shares > 0]

    if not active_tickers:
        st.info("目前尚無持有庫存。點擊上方「⚖️ 新增交易」開始第一筆買進！")
        return

    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([2, 1.5, 1.5, 2, 2, 2])
    hc1.markdown("**股票代號**")
    hc2.markdown("**庫存 (股)**")
    hc3.markdown("**均價**")
    hc4.markdown("**市值**")
    hc5.markdown("**未實現損益**")
    hc6.markdown("**操作**")
    st.divider()

    name_map = get_tw_stock_mapping()
    for ticker in sorted(active_tickers):
        pos_obj = account.get_position(ticker)

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
                if st.button("交易", key=f"t_{ticker}", use_container_width=True):
                    trade_dialog(db_manager, prefill_ticker=ticker)
            with btn_col2:
                if st.button("歷史", key=f"h_{ticker}", use_container_width=True):
                    history_dialog(ticker)

        st.markdown("<hr style='margin: 0; opacity: 0.2;'>", unsafe_allow_html=True)