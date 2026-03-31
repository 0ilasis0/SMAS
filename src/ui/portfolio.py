import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from bt.params import TaxRate
from data.const import StockCol
from ui.const import EncodingConst, PortfolioCol

PORTFOLIO_FILE = "data/portfolio.json"

# ==========================================
# 1. 資料層：極致防呆的讀寫邏輯
# ==========================================
def get_default_portfolio() -> dict:
    return {
        PortfolioCol.GLOBAL_CASH: 2000000.0,
        PortfolioCol.POSITIONS: {}
    }

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding=EncodingConst.UTF8) as f:
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
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, "w", encoding=EncodingConst.UTF8) as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"❌ 資金檔存檔失敗: {e}")

# ==========================================
# 2. 交易引擎：彈出式手動買賣視窗 (支援真實稅費結算)
# ==========================================
@st.dialog("⚖️ 交易與明細登錄")
def trade_dialog(db_manager, prefill_ticker: str = ""):
    pf = st.session_state.portfolio

    # 1. 股票代號輸入
    raw_ticker = st.text_input("🔍 股票代號 (輸入後按 Enter 抓取現價)", value=prefill_ticker, placeholder="例如: 2337.TW")

    if not raw_ticker:
        return

    ticker = raw_ticker.strip().upper()
    if not ticker.endswith(".TW") and not ticker.endswith(".TWO"):
        ticker += ".TW"

    if ticker not in pf[PortfolioCol.POSITIONS]:
        pf[PortfolioCol.POSITIONS][ticker] = {
            PortfolioCol.SHARES: 0,
            PortfolioCol.AVG_COST: 0.0,
            PortfolioCol.HISTORY: []
        }

    pos_data = pf[PortfolioCol.POSITIONS][ticker]
    current_cash = pf[PortfolioCol.GLOBAL_CASH]
    current_shares = pos_data[PortfolioCol.SHARES]
    current_avg_cost = pos_data[PortfolioCol.AVG_COST]

    # 動態抓取最新股價
    fetched_price = current_avg_cost if current_avg_cost > 0 else 10.0
    if db_manager:
        try:
            df = db_manager.get_daily_data(ticker)
            if not df.empty:
                fetched_price = float(df[StockCol.CLOSE].iloc[-1])
        except Exception:
            pass

    st.markdown("---")

    # 顯示歷史紀錄 (支援新舊欄位防呆)
    if pos_data[PortfolioCol.HISTORY]:
        st.caption(f"📜 {ticker} 歷史交易紀錄")
        df_history = pd.DataFrame(pos_data[PortfolioCol.HISTORY])

        # 🚀 向下相容：如果舊資料沒有 fee 跟 tax 欄位，自動補 0
        if "fee" not in df_history.columns:
            df_history["fee"] = 0
        if "tax" not in df_history.columns:
            df_history["tax"] = 0

        df_history = df_history[["date", "action", "price", "shares", "fee", "tax", "total"]]
        df_history.columns = ["時間", "動作", "單價", "股數", "手續費", "交易稅", "交割淨額"]
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info(f"尚無 {ticker} 的交易紀錄。")

    st.markdown("---")

    # 2. UI 交易輸入區
    action = st.radio("交易動作", ["🟢 買進 (BUY)", "🔴 賣出 (SELL)"], horizontal=True)
    is_buy = action.startswith("🟢")

    col1, col2 = st.columns(2)
    with col1:
        trade_price = st.number_input("成交單價 (元)", min_value=0.01, value=fetched_price, step=0.5, format="%.2f")
    with col2:
        trade_shares = st.number_input("成交股數 (股)", min_value=1, value=1000, step=1000)

    # ==========================================
    # 🚀 升級：台灣股市真實稅費計算引擎
    # ==========================================
    base_amount = trade_price * trade_shares
    # 手續費：取計算值與低消的最大值，台股習慣以整數計 (無條件捨去或四捨五入皆可，此處用 int)
    fee = int(max(TaxRate.MIN_FEE, base_amount * TaxRate.FEE_RATE))
    # 交易稅：僅賣出時收取
    tax = int(base_amount * TaxRate.TAX_RATE) if not is_buy else 0

    if is_buy:
        total_settlement = base_amount + fee
        st.info(f"💵 預估應付交割：**${total_settlement:,.0f}** (含手續費 ${fee})")
    else:
        total_settlement = base_amount - fee - tax
        st.info(f"💵 預估應收交割：**${total_settlement:,.0f}** (扣除手續費 ${fee}、交易稅 ${tax})")

    # 3. 送出與結算邏輯
    if st.button("確認送出交易", type="primary", use_container_width=True):

        # --- 買進防呆與邏輯 ---
        if is_buy:
            if total_settlement > current_cash:
                st.error(f"❌ 可用現金不足！(應付: ${total_settlement:,.0f} / 餘額: ${current_cash:,.0f})")
                return

            # 將手續費直接「攤提」進成本，符合券商真實均價算法
            old_total_cost = current_shares * current_avg_cost
            new_total_cost = old_total_cost + total_settlement
            new_shares = current_shares + trade_shares
            new_avg_cost = new_total_cost / new_shares

            pf[PortfolioCol.GLOBAL_CASH] -= total_settlement
            pos_data[PortfolioCol.SHARES] = new_shares
            pos_data[PortfolioCol.AVG_COST] = new_avg_cost

        # --- 賣出防呆與邏輯 ---
        else:
            if trade_shares > current_shares:
                st.error(f"❌ 庫存餘額不足！(目前僅持有: {current_shares} 股)")
                return

            new_shares = current_shares - trade_shares
            new_avg_cost = current_avg_cost if new_shares > 0 else 0.0

            pf[PortfolioCol.GLOBAL_CASH] += total_settlement
            pos_data[PortfolioCol.SHARES] = new_shares
            pos_data[PortfolioCol.AVG_COST] = new_avg_cost

        # 寫入歷史紀錄 (包含稅費細節)
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

@st.dialog("🏦 存取資金")
def cash_operation_dialog():
    st.markdown("請輸入您要存入或提出的金額。")
    pf = st.session_state.portfolio

    op_type = st.radio("操作類型", ["📥 存入資金 (Deposit)", "📤 提出資金 (Withdraw)"], horizontal=True)
    amount = st.number_input("金額 (NTD)", min_value=0.0, max_value=100000000.0, step=10000.0, format="%.0f")

    if st.button("確認執行", type="primary", use_container_width=True):
        if op_type.startswith("📤"):
            if amount > pf[PortfolioCol.GLOBAL_CASH]:
                st.error("❌ 餘額不足！無法提出大於目前帳戶可用現金的金額。")
                return
            pf[PortfolioCol.GLOBAL_CASH] -= amount
            st.success(f"✅ 成功提出 {amount:,.0f} 元。")
        else:
            pf[PortfolioCol.GLOBAL_CASH] += amount
            st.success(f"✅ 成功存入 {amount:,.0f} 元。")

        save_portfolio(pf)
        st.rerun()

# ==========================================
# 3. 視圖層：資產管理中心主畫面
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
            if st.button("🏦 存提款", use_container_width=True):
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

    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([2, 1.5, 1.5, 2, 2, 1.5])
    hc1.markdown("**股票代號**")
    hc2.markdown("**庫存 (股)**")
    hc3.markdown("**均價**")
    hc4.markdown("**市值**")
    hc5.markdown("**未實現損益**")
    hc6.markdown("**操作**")
    st.divider()

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

        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns([2, 1.5, 1.5, 2, 2, 1.5])

        rc1.markdown(f"**{ticker}**")
        rc2.markdown(f"{shares:,}")
        rc3.markdown(f"${avg_cost:,.2f}")
        rc4.markdown(f"${market_val:,.0f}")
        rc5.markdown(pnl_text, unsafe_allow_html=True)

        with rc6:
            if st.button("明細/交易", key=f"table_trade_{ticker}", use_container_width=True):
                trade_dialog(db_manager, prefill_ticker=ticker)

        st.markdown("<hr style='margin: 0; opacity: 0.2;'>", unsafe_allow_html=True)
