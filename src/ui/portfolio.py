import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from bt.params import TaxRate
from data.const import StockCol
from path import PathConfig
from ui.const import EncodingConst, PortfolioCol
from ui.stock_names import get_tw_stock_mapping


# ==========================================
# 1. 資料層：極致防呆的讀寫邏輯
# ==========================================
def get_default_portfolio() -> dict:
    return {
        PortfolioCol.GLOBAL_CASH: 0.0,
        PortfolioCol.POSITIONS: {}
    }
def load_portfolio() -> dict:
    if os.path.exists(PathConfig.PORTFOLIO):
        try:
            with open(PathConfig.PORTFOLIO, "r", encoding=EncodingConst.UTF8) as f:
                data = json.load(f)
                if PortfolioCol.GLOBAL_CASH not in data or PortfolioCol.POSITIONS not in data:
                    return get_default_portfolio()
                for ticker, pos in data[PortfolioCol.POSITIONS].items():
                    if PortfolioCol.HISTORY not in pos:
                        pos[PortfolioCol.HISTORY] = []
                return data
        except Exception:
            return get_default_portfolio()
    return get_default_portfolio()

def save_portfolio(portfolio_data: dict):
    try:
        os.makedirs(os.path.dirname(PathConfig.PORTFOLIO), exist_ok=True)
        with open(PathConfig.PORTFOLIO, "w", encoding=EncodingConst.UTF8) as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"❌ 資金檔存檔失敗: {e}")

def recalculate_position(history: list) -> tuple[int, float]:
    """
    動態帳務重算引擎：
    根據歷史紀錄，由頭到尾重算真實的「目前庫存」與「加權平均成本」。
    """
    current_shares = 0
    current_total_cost = 0.0

    for r in history:
        action = r.get("action", "BUY")
        shares = int(r.get("shares", 0))
        total = float(r.get("total", 0.0))

        if action == "BUY":
            current_shares += shares
            current_total_cost += total
        elif action == "SELL":
            if current_shares > 0:
                avg_cost = current_total_cost / current_shares
                current_shares -= shares
                if current_shares <= 0:
                    current_shares = 0
                    current_total_cost = 0.0
                else:
                    current_total_cost = avg_cost * current_shares

    avg_cost = current_total_cost / current_shares if current_shares > 0 else 0.0
    return current_shares, avg_cost

def get_active_buys(history: list) -> list:
    """找出構成「當前庫存均價」的所有有效買進批次"""
    temp_shares = 0
    last_zero_idx = -1

    # 1. 找出最後一次「空手」的時間點
    for i, r in enumerate(history):
        if r.get("action") == "BUY":
            temp_shares += int(r.get("shares", 0))
        else:
            temp_shares -= int(r.get("shares", 0))
            if temp_shares <= 0:
                temp_shares = 0
                last_zero_idx = i

    # 2. 擷取在最後一次空手「之後」的所有 BUY 紀錄
    active_buys = []
    for i in range(last_zero_idx + 1, len(history)):
        r = history[i]
        if r.get("action") == "BUY":
            active_buys.append(r)

    return active_buys

# ==========================================
# 2. UI 層：歷史資料編輯器 (新增)
# ==========================================
@st.dialog("📜 歷史資料與帳務編修", width="large")
def history_dialog(ticker: str):
    st.markdown(f"管理 **{ticker}** 的歷史交易紀錄。您可以直接在表格中點選該列並按 `Delete` 來刪除錯誤的紀錄。")
    st.warning("⚠️ 刪除歷史紀錄將會**自動重新計算**您的庫存數量與平均成本，但**不會**退還/扣除可用現金。若需調整資金請至「存取資金」操作。")

    pf = st.session_state.portfolio
    pos_data = pf[PortfolioCol.POSITIONS].get(ticker, {PortfolioCol.HISTORY: []})
    history = pos_data[PortfolioCol.HISTORY]

    if not history:
        st.info("尚無任何歷史紀錄。")
        return

    # 將歷史紀錄轉換為 DataFrame 以供編輯
    df = pd.DataFrame(history)
    # 確保舊資料格式相容
    for col in ["fee", "tax"]:
        if col not in df.columns:
            df[col] = 0

    # 整理顯示順序與欄位名稱
    df_display = df[["date", "action", "price", "shares", "fee", "tax", "total"]].copy()
    df_display.columns = ["時間", "動作", "單價", "股數", "手續費", "交易稅", "交割淨額"]

    # 使用 st.data_editor 讓使用者可以直接刪除或修改列 (num_rows="dynamic" 允許刪除)
    edited_df = st.data_editor(df_display, num_rows="dynamic", use_container_width=True, hide_index=True)

    if st.button("💾 儲存歷史變更並重新結算庫存", type="primary", use_container_width=True):
        # 1. 將編輯後的 DataFrame 還原回系統的字典格式
        new_history = []
        for _, row in edited_df.iterrows():
            new_history.append({
                "date": str(row["時間"]),
                "action": str(row["動作"]),
                "price": float(row["單價"]),
                "shares": int(row["股數"]),
                "fee": int(row["手續費"]),
                "tax": int(row["交易稅"]),
                "total": float(row["交割淨額"])
            })

        # 2. 使用核心引擎重新計算正確的股數與均價
        new_shares, new_avg_cost = recalculate_position(new_history)

        # 3. 寫入系統狀態
        pf[PortfolioCol.POSITIONS][ticker][PortfolioCol.HISTORY] = new_history
        pf[PortfolioCol.POSITIONS][ticker][PortfolioCol.SHARES] = new_shares
        pf[PortfolioCol.POSITIONS][ticker][PortfolioCol.AVG_COST] = new_avg_cost

        save_portfolio(pf)
        st.success(f"✅ 重算完成！目前庫存: {new_shares} 股 / 均價: {new_avg_cost:.2f} 元")
        st.rerun()

# ==========================================
# 3. 交易引擎：彈出式手動買賣視窗
# ==========================================
@st.dialog("⚖️ 新增交易", width="large")
def trade_dialog(db_manager, prefill_ticker: str = "", prefill_action: str = None, prefill_price: float = None, prefill_shares: int = None):
    pf = st.session_state.portfolio

    raw_ticker = st.text_input("🔍 股票代號 (輸入後按 Enter 抓取現價)", value=prefill_ticker, placeholder="例如: 2330.TW")
    if not raw_ticker: return

    ticker = raw_ticker.strip().upper()
    if not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        ticker += ".TW"

    if ticker not in pf[PortfolioCol.POSITIONS]:
        pf[PortfolioCol.POSITIONS][ticker] = {PortfolioCol.SHARES: 0, PortfolioCol.AVG_COST: 0.0, PortfolioCol.HISTORY: []}

    pos_data = pf[PortfolioCol.POSITIONS][ticker]
    current_cash = pf[PortfolioCol.GLOBAL_CASH]
    current_shares = pos_data[PortfolioCol.SHARES]
    current_avg_cost = pos_data[PortfolioCol.AVG_COST]

    # 🚀 動態抓取股價邏輯升級：若 AI 有傳入預期觸價，優先使用 AI 的價格
    fetched_price = current_avg_cost if current_avg_cost > 0 else 10.0
    if prefill_price is not None and prefill_price > 0:
        fetched_price = float(prefill_price)
    elif db_manager:
        try:
            df = db_manager.get_daily_data(ticker)
            if not df.empty:
                fetched_price = float(df[StockCol.CLOSE].iloc[-1])
        except Exception:
            pass

    st.markdown("---")
    active_buys = get_active_buys(pos_data[PortfolioCol.HISTORY])
    if active_buys and current_shares > 0:
        st.caption(f"💡 目前持股均價 **${current_avg_cost:.2f}** 的構成批次 (有效買進明細)：")
        df_buys = pd.DataFrame(active_buys)
        df_buys = df_buys[["date", "price", "shares", "total"]]
        df_buys.columns = ["買進時間", "買進單價", "股數", "含費總成本"]
        st.dataframe(df_buys, use_container_width=True, hide_index=True)
    elif current_shares <= 0:
        st.info("目前無庫存。")

    st.markdown("---")

    # 🚀 自動切換買賣按鈕狀態
    default_action_idx = 0
    if prefill_action == "SELL":
        default_action_idx = 1

    action = st.radio("交易動作", ["🟢 買進 (BUY)", "🔴 賣出 (SELL)"], index=default_action_idx, horizontal=True)
    is_buy = action.startswith("🟢")

    col1, col2 = st.columns(2)
    with col1:
        trade_price = st.number_input("成交單價 (元)", min_value=0.01, value=fetched_price, step=0.5, format="%.2f")
    with col2:
        # 🚀 載入 AI 建議的股數，並加上防呆 (若建議賣出但庫存不足，自動降至庫存最大值)
        default_shares = prefill_shares if prefill_shares is not None else 1000
        if default_shares <= 0:
            default_shares = 1000
        if not is_buy and default_shares > current_shares and current_shares > 0:
            default_shares = current_shares

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

    if st.button("確認送出交易", type="primary", use_container_width=True):
        if is_buy:
            if total_settlement > current_cash:
                st.error(f"❌ 可用現金不足！(應付: ${total_settlement:,.0f} / 餘額: ${current_cash:,.0f})")
                return
            old_total_cost = current_shares * current_avg_cost
            new_total_cost = old_total_cost + total_settlement
            new_shares = current_shares + trade_shares
            new_avg_cost = new_total_cost / new_shares
            pf[PortfolioCol.GLOBAL_CASH] -= total_settlement

        else:
            if trade_shares > current_shares:
                st.error(f"❌ 庫存餘額不足！(目前僅持有: {current_shares} 股)")
                return
            new_shares = current_shares - trade_shares
            new_avg_cost = current_avg_cost if new_shares > 0 else 0.0
            pf[PortfolioCol.GLOBAL_CASH] += total_settlement

        pos_data[PortfolioCol.SHARES] = new_shares
        pos_data[PortfolioCol.AVG_COST] = new_avg_cost

        trade_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": "BUY" if is_buy else "SELL",
            "price": trade_price,
            "shares": trade_shares,
            "fee": fee,
            "tax": tax,
            "total": total_settlement
        }
        pos_data[PortfolioCol.HISTORY].append(trade_record)

        save_portfolio(pf)
        st.rerun()

# 存取資金 Dialog
@st.dialog("🏦 存取資金")
def cash_operation_dialog():
    st.markdown("請輸入您要存入或提出的金額。")
    pf = st.session_state.portfolio
    op_type = st.radio("操作類型", ["📥 存入資金 (Deposit)", "📤 提出資金 (Withdraw)"], horizontal=True)
    amount = st.number_input("金額 (NTD)", min_value=0.0, max_value=100000000.0, step=10000.0, format="%.0f")

    if st.button("確認執行", type="primary", use_container_width=True):
        if op_type.startswith("📤"):
            if amount > pf[PortfolioCol.GLOBAL_CASH]:
                st.error("❌ 餘額不足！")
                return
            pf[PortfolioCol.GLOBAL_CASH] -= amount
            st.success(f"✅ 成功提出 {amount:,.0f} 元。")
        else:
            pf[PortfolioCol.GLOBAL_CASH] += amount
            st.success(f"✅ 成功存入 {amount:,.0f} 元。")
        save_portfolio(pf)
        st.rerun()

# ==========================================
# 4. 視圖層：資產管理中心主畫面
# ==========================================
def render_portfolio_page(db_manager=None):
    st.title("💼 資產管理中心")
    st.markdown("---")

    pf = st.session_state.portfolio

    total_market_value = 0.0
    total_cost_value = 0.0
    current_prices = {}

    for ticker, pos_data in pf[PortfolioCol.POSITIONS].items():
        shares = pos_data.get(PortfolioCol.SHARES, 0)
        avg_cost = pos_data.get(PortfolioCol.AVG_COST, 0.0)

        if shares > 0:
            current_price = avg_cost
            if db_manager:
                try:
                    df_latest = db_manager.get_daily_data(ticker)
                    if not df_latest.empty:
                        current_price = float(df_latest[StockCol.CLOSE].iloc[-1])
                except Exception:
                    pass

            current_prices[ticker] = current_price
            total_market_value += (shares * current_price)
            total_cost_value += (shares * avg_cost)

    total_assets = pf[PortfolioCol.GLOBAL_CASH] + total_market_value

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1.5])
    col1.metric("👑 預估總資產", f"${total_assets:,.0f}")
    col2.metric("🎯 總成本", f"${total_cost_value:,.0f}")
    col3.metric("💵 可用現金", f"${pf[PortfolioCol.GLOBAL_CASH]:,.0f}")

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
    active_positions = {t: p for t, p in pf[PortfolioCol.POSITIONS].items() if p.get(PortfolioCol.SHARES, 0) > 0}

    if not active_positions:
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
    for ticker in sorted(active_positions.keys()):
        pos_data = active_positions[ticker]
        shares = pos_data[PortfolioCol.SHARES]
        avg_cost = pos_data[PortfolioCol.AVG_COST]
        c_price = current_prices.get(ticker, avg_cost)

        market_val = shares * c_price
        cost_val = shares * avg_cost
        pnl = market_val - cost_val
        pnl_pct = (pnl / cost_val) if cost_val > 0 else 0.0

        pnl_color = "#00cc66" if pnl >= 0 else "#ff4b4b"
        pnl_text = f"<span style='color: {pnl_color}; font-weight: bold;'>${pnl:,.0f} ({pnl_pct:.2%})</span>"
        ch_name = name_map.get(ticker, "")

        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns([2, 1.5, 1.5, 2, 2, 2])

        rc1.markdown(f"**{ticker}** <br> <span style='font-size:0.85em; color:gray;'>{ch_name}</span>", unsafe_allow_html=True)
        rc2.markdown(f"{shares:,}")
        rc3.markdown(f"${avg_cost:,.2f}")
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
