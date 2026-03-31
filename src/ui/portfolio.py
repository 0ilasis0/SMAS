import json
import os

import streamlit as st

from path import PathConfig
from ui.const import EncodingConst, PortfolioCol
from ui.params import AcountLimit


# ==========================================
# 1. 資料層：極致防呆的讀寫邏輯
# ==========================================
def get_default_portfolio() -> dict:
    """提供標準的預設資金結構"""
    return {
        PortfolioCol.GLOBAL_CASH: 0.0,
        PortfolioCol.POSITIONS: {}
    }

def load_portfolio() -> dict:
    if os.path.exists(PathConfig.PORTFOLIO):
        try:
            with open(PathConfig.PORTFOLIO, "r", encoding=EncodingConst.STD_FONT) as f:
                data = json.load(f)
                if PortfolioCol.GLOBAL_CASH not in data or PortfolioCol.POSITIONS not in data:
                    return get_default_portfolio()
                # 自動補齊舊版資料缺少的 history 欄位
                for ticker, pos in data[PortfolioCol.POSITIONS].items():
                    if PortfolioCol.HISTORY not in pos:
                        pos[PortfolioCol.HISTORY] = []
                return data
        except Exception:
            return get_default_portfolio()
    return get_default_portfolio()

def save_portfolio(portfolio_data: dict):
    """安全寫入本地資金檔"""
    try:
        os.makedirs(os.path.dirname(PathConfig.PORTFOLIO), exist_ok=True)
        with open(PathConfig.PORTFOLIO, "w", encoding=EncodingConst.STD_FONT) as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"❌ 資金檔存檔失敗: {e}")

# ==========================================
# 2. 功能層：存取資金防呆彈窗
# ==========================================
@st.dialog("🏦 存取資金")
def cash_operation_dialog():
    st.markdown("請輸入您要存入或提出的金額。")
    pf = st.session_state.portfolio

    op_type = st.radio("操作類型", ["📥 存入資金 (Deposit)", "📤 提出資金 (Withdraw)"], horizontal=True)
    amount = st.number_input("金額 (NTD)", min_value=AcountLimit.MIN_MONEY, max_value=AcountLimit.MAX_MONEY, step=AcountLimit.STEP_MONEY, format="%.0f")

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
    """渲染獨立的資產管理頁面"""
    st.title("💼 資產管理中心")
    st.markdown("---")

    pf = st.session_state.portfolio

    # --- 計算總資產 (現金 + 所有股票市值) ---
    total_market_value = 0.0
    from data.const import StockCol

    for ticker, pos_data in pf[PortfolioCol.POSITIONS].items():
        shares = pos_data.get(PortfolioCol.SHARES, 0)
        if shares > 0:
            # 動態抓取最新股價 (若無 db 則預設用成本價估算)
            current_price = pos_data.get(PortfolioCol.AVG_COST, 0.0)
            if db_manager:
                try:
                    df_latest = db_manager.get_daily_data(ticker)
                    if not df_latest.empty:
                        current_price = df_latest[StockCol.CLOSE].iloc[-1]
                except Exception:
                    pass
            total_market_value += (shares * current_price)

    total_assets = pf[PortfolioCol.GLOBAL_CASH] + total_market_value

    # --- 頂部：總資產看板 ---
    col1, col2, col3 = st.columns([2, 2, 1])
    col1.metric("👑 預估總資產 (Total Assets)", f"${total_assets:,.0f}")
    col2.metric("💵 可用現金 (Purchasing Power)", f"${pf[PortfolioCol.GLOBAL_CASH]:,.0f}")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏦 存取資金", use_container_width=True):
            cash_operation_dialog()

    st.markdown("---")

    # --- 底部：庫存詳情預留區 (Phase 2 將會在這裡實作) ---
    st.subheader("📦 目前庫存明細")
    st.info("💡 買賣手動輸入引擎與個別庫存明細展開功能，將於 Phase 2 部署。")
