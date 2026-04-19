import sys
import warnings

if 'warnings' not in sys.modules:
    sys.modules['warnings'] = warnings

import time
from typing import TYPE_CHECKING

import streamlit as st

from bt.const import TradeDecision
from controller import IDSSController
from data.fetcher import Fetcher
from data.manager import DataManager
from data.updater import DataUpdater
from ml.engine import QuantAIEngine
from ui.backtest import render_backtest_tab
from ui.chart import render_chart
from ui.const import APIKey, Page, SessionKey
from ui.params import BacktestParams
from ui.portfolio import load_portfolio, render_portfolio_page, trade_dialog
from ui.report import render_report
from ui.sidebar import render_sidebar
from ui.state import init_session_state

if TYPE_CHECKING:
    from bt.account import Account

HAS_AUTO_UPDATED_KEY = "has_auto_updated"


def sync_market_data(ticker: str, force_wipe: bool = False, force_sync: bool = False):
    """獨立的資料同步管線：負責抓取個股、大盤與企業事件 (法說會/除權息)"""
    db = DataManager()
    fetcher = Fetcher()
    updater = DataUpdater(db, fetcher)
    updater.update_market_data(ticker=ticker, force_wipe=force_wipe, force_sync=force_sync)


def run_mlops_pipeline(ticker: str):
    """執行完整的雙軌訓練管線，並在訓練前後開關 UI 鎖"""
    try:
        with st.status(f"🚀 MLOps 執行中: {ticker}", expanded=True) as status:
            try:
                # 階段 1：更新資料
                st.write("📥 正在從 Yahoo Finance 重新下載最新歷史與大盤雙軌資料...")
                sync_market_data(ticker, force_wipe=True)
                engine_bt = QuantAIEngine(ticker=ticker, oos_days=BacktestParams.MAX_DAYS)

                # 階段 2：訓練回測大腦
                st.write(f"🧠 正在訓練回測大腦 (OOS={BacktestParams.MAX_DAYS})... 這將花費數分鐘")
                engine_bt.train_all_models(save_models=True)

                # 階段 3：訓練實盤大腦
                st.write("⚔️ 正在訓練實盤大腦 (OOS=0)...")
                engine_live = QuantAIEngine(ticker=ticker, oos_days=0)
                engine_live.train_all_models(save_models=True)

                status.update(label="✅ 訓練完畢！準備重新載入系統...", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ 訓練過程中發生錯誤", state="error", expanded=True)
                st.error(f"詳細錯誤: {e}")
                if st.button("🔙 返回控制台"):
                    st.session_state[SessionKey.IS_TRAINING.value] = False
                    st.rerun()
                st.stop()
    finally:
        st.session_state[SessionKey.IS_TRAINING.value] = False

    st.rerun()


def run_global_mlops_pipeline():
    """執行全域 MLOps：走訪自選股清單，全部重新抓資料並所有模型重新訓練"""
    # 從所有組合包中提取出不重複的股票清單 (聯集)
    account: "Account" = st.session_state.get(SessionKey.PORTFOLIO.value)
    if not account: return

    all_tickers = set()
    for sp in account.sub_portfolios.values():
        all_tickers.update(sp.watch_tickers)

    watch_list = sorted(list(all_tickers))
    total_stocks = len(watch_list)

    if not watch_list:
        st.warning("⚠️ 所有組合包皆無關注標的，無需訓練。")
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = False
        time.sleep(1)
        st.rerun()
        return

    progress_bar = st.progress(0, text="準備啟動 MLOps 全域管線...")

    try:
        with st.status("🔄 全域模型訓練執行中 (週末 MLOps 管線)...", expanded=True) as status:
            for idx, ticker in enumerate(watch_list):
                current_step = idx + 1

                # 動態更新進度條文字
                progress_bar.progress(idx / total_stocks, text=f"進度：正在訓練 {ticker} ({current_step}/{total_stocks})")
                status.update(label=f"🔄 正在處理 {ticker} ({current_step}/{total_stocks})...")

                try:
                    # 階段 1：清除舊資料，下載最新 K 線
                    st.write(f"📥 [{ticker}] 清除快取並下載最新資料...")
                    sync_market_data(ticker, force_wipe=True)
                    engine_bt = QuantAIEngine(ticker=ticker, oos_days=BacktestParams.MAX_DAYS)

                    # 階段 2：訓練回測大腦
                    st.write(f"🧠 [{ticker}] 訓練歷史回測大腦...")
                    engine_bt.train_all_models(save_models=True)

                    # 階段 3：訓練實盤大腦
                    st.write(f"⚔️ [{ticker}] 訓練最新實盤大腦...")
                    engine_live = QuantAIEngine(ticker=ticker, oos_days=0)
                    engine_live.train_all_models(save_models=True)

                    st.write(f"✅ [{ticker}] 模型升級完畢！")
                except Exception as e:
                    st.error(f"❌ [{ticker}] 訓練失敗: {e}")

            # 完成後將進度條推滿
            progress_bar.progress(1.0, text="🎉 全域模型訓練完畢！")
            status.update(label="🎉 所有自選股模型訓練完畢！系統已吸收最新市場趨勢。", state="complete", expanded=False)
            time.sleep(2)
            progress_bar.empty()
    finally:
        st.session_state[SessionKey.IS_GLOBAL_TRAINING.value] = False

    st.rerun()

# ==========================================
# 主程式流 (Main Application Flow)
# ==========================================
def main():
    init_session_state()
    selected_persona = render_sidebar()

    if not st.session_state.get(HAS_AUTO_UPDATED_KEY, False):
        st.toast("🔄 系統首次啟動：正在檢查並同步最新市場收盤資料...", icon="⏳")
        status_placeholder = st.empty()
        has_error = False
        with status_placeholder.container():
            with st.status("🔄 系統首次啟動：同步所有追蹤標的...", expanded=True) as status:

                # 🌟 改為從所有組合包收集股票
                account = st.session_state.get(SessionKey.PORTFOLIO.value)
                all_tickers = set()
                if account:
                    for sp in account.sub_portfolios.values():
                        all_tickers.update(sp.watch_tickers)
                watch_list = sorted(list(all_tickers))

                for t in watch_list:
                    st.write(f"📥 正在檢查/同步 {t} 的最新 K 線與大盤資料...")
                    try:
                        sync_market_data(t)
                    except Exception as e:
                        st.write(f"⚠️ {t} 同步失敗: {e}")
                        has_error = True

                if not has_error:
                    status.update(label="✅ 所有市場資料已校準！", state="complete", expanded=False)
                    st.session_state[HAS_AUTO_UPDATED_KEY] = True
                else:
                    status.update(label="❌ 部分資料同步失敗，請檢查下方訊息", state="error", expanded=True)

                st.session_state[HAS_AUTO_UPDATED_KEY] = True
                status.update(label="✅ 所有市場資料已校準至最新交易日！", state="complete", expanded=False)

        if not has_error:
            time.sleep(1)
            status_placeholder.empty()

    current_ticker = st.session_state.get(SessionKey.CURRENT_TICKER.value)
    current_page = st.session_state.get(SessionKey.CURRENT_PAGE.value)

    if current_page == Page.PORTFOLIO.value:
        # 如果大腦已經在，就用它的 db；如果沒有，我們臨時建一個給它查現價
        ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)
        db_manager = ctrl_live.engine.db if ctrl_live else None

        if db_manager is None:
            from data.manager import DataManager
            db_manager = DataManager()

        render_portfolio_page(db_manager)
        return  # 渲染完資產管理就直接結束，不往下跑決策大廳的邏輯

    # ==========================================
    # 進入 IDSS 決策大廳 (Dashboard)
    # ==========================================
    account = st.session_state.get(SessionKey.PORTFOLIO.value, load_portfolio())
    current_sp_name = st.session_state.get("CURRENT_SUB_PORTFOLIO")
    if not current_sp_name or current_sp_name not in account.sub_portfolios:
        st.warning("👈 尚未選擇投資組合包，請由左側邊欄選擇或建立。")
        st.stop()

    current_sp = account.get_sub_portfolio(current_sp_name)
    current_ticker = st.session_state.get(SessionKey.CURRENT_TICKER.value)

    active_tickers = list(set(current_sp.watch_tickers) | set(current_sp.positions.keys()))

    if current_ticker not in active_tickers:
        if active_tickers:
            # 如果組合包有股票 (不管是自選還是庫存)，強制切換成第一檔
            current_ticker = sorted(active_tickers)[0]
            st.session_state[SessionKey.CURRENT_TICKER.value] = current_ticker
            st.session_state[SessionKey.CTRL_LIVE.value] = None

            # 如果這檔股票有庫存，卻不在自選單裡，順手幫您補回去並存檔
            if current_ticker not in current_sp.watch_tickers:
                current_sp.watch_tickers.append(current_ticker)
                from ui.portfolio import save_portfolio
                save_portfolio(account)
        else:
            # 如果組合包真的完全沒有自選股也沒有庫存，強制變成 None
            current_ticker = None
            st.session_state[SessionKey.CURRENT_TICKER.value] = None
            st.session_state[SessionKey.CTRL_LIVE.value] = None

    if current_ticker is None:
        st.warning(f"👈 您目前的組合包 **【{current_sp.name}】** 內尚無關注標的，請先從左側邊欄新增股票！")
        st.stop()

    if st.session_state.get(SessionKey.IS_GLOBAL_TRAINING.value, False):
        run_global_mlops_pipeline()
        st.stop()

    if st.session_state.get(SessionKey.IS_TRAINING.value, False):
        run_mlops_pipeline(current_ticker)
        st.stop()

    # 確保「實盤大腦」載入
    if st.session_state.get(SessionKey.CTRL_LIVE.value) is None:
        ctrl = IDSSController(ticker=current_ticker, oos_days=0)

        if ctrl.load_system():
            st.session_state[SessionKey.CTRL_LIVE.value] = ctrl
            st.toast(f"{current_ticker} 實盤引擎就緒！", icon="🟢")
        else:
            st.warning(f"⚠️ 系統偵測到 **{current_ticker}** 缺乏完整的 AI 模型權重檔。")
            if st.button("🚀 立即啟動 AI 訓練管線", type="primary"):
                st.session_state[SessionKey.IS_TRAINING.value] = True
                st.rerun()
            st.stop()

    # 決定「可動用資金 (Usable Cash)」
    if current_sp.use_shared_cash:
        usable_cash = account.unallocated_cash  # 使用系統未分配流動資金
    else:
        usable_cash = current_sp.allocated_cash # 使用該組合包的專屬小水桶

    # 從「當前組合包」提取該股票的部位狀態
    pos_obj = current_sp.get_position(current_ticker)
    my_shares = pos_obj.shares
    my_avg_cost = pos_obj.avg_cost

    st.markdown("---")

    title_col, btn_col = st.columns([3, 1])
    with title_col:
        st.title(f"📊 IDSS 決策大廳 - {current_ticker}")
        st.caption(f"📂 目前操作組合包：**【{current_sp.name}】**")

    with btn_col:
        st.write("") # 往下擠一點對齊標題
        if st.button("🔄 盤中同步現價", use_container_width=True):
            with st.spinner("正在從 Yahoo 同步所有關注標的..."):
                all_tickers = set()
                for sp in account.sub_portfolios.values():
                    all_tickers.update(sp.watch_tickers)

                for t in all_tickers:
                    try:
                        sync_market_data(t, force_sync=True)
                    except Exception as e:
                        st.error(f"⚠️ {t} 同步失敗: {e}")

                st.session_state[HAS_AUTO_UPDATED_KEY] = False # 讓系統重整後知道要重讀資料
                st.toast("✅ 盤中即時資料已強制覆蓋更新！", icon="⚡")
                time.sleep(1)
                st.rerun()
    st.markdown("---")

    render_chart()

    tab1, tab2 = st.tabs(["🎯 今日行動指令", "⏱️ IDSS 歷史回測模擬"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        is_training = st.session_state.get(SessionKey.IS_TRAINING.value, False)

        if my_shares > 0:
            st.info(f"💼 **目前庫存**：持有 {my_shares:,} 股，均價 {my_avg_cost:,.2f} 元 (佔系統總資產 {pos_obj.market_value / account.total_equity:.1%} )")
        else:
            # 顯示當前可動用的真實資金
            cash_source_str = "系統活資金" if current_sp.use_shared_cash else "組合包專屬資金"
            st.info(f"💼 **目前庫存**：空手觀望中。可用 {cash_source_str}：{usable_cash:,.0f} 元")

        if st.button("🚀 產生今日 AI 決策與戰報", type="primary", use_container_width=True, disabled=is_training):
            with st.spinner("神經網路推論中，正在呼叫 Gemini 分析市場新聞空氣..."):
                ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)

                # 4. 餵給 AI 的資金變成 usable_cash
                result = ctrl_live.execute_decision(
                    available_cash=usable_cash,
                    current_position=my_shares,
                    avg_cost=my_avg_cost,
                    persona=selected_persona
                )
                st.session_state[SessionKey.LAST_RESULT.value] = result

        last_result = st.session_state.get(SessionKey.LAST_RESULT.value)
        if last_result is not None:
            render_report(last_result)

            # ==========================================
            # AI 決策與真實帳房的「一鍵執行橋樑」
            # ==========================================
            decision = last_result.get(APIKey.DECISION.value, {})
            action = decision.get(APIKey.ACTION.value, TradeDecision.HOLD.value)

            if action in [TradeDecision.BUY.value, TradeDecision.SELL.value]:
                st.markdown("---")
                btn_icon = "🛒" if action == TradeDecision.BUY.value else "💸"
                btn_text = f"{btn_icon} 採納 AI 建議，立即執行 {action.upper()} 交易"

                if st.button(btn_text, type="primary", use_container_width=True, disabled=is_training):
                    ctrl_live = st.session_state.get(SessionKey.CTRL_LIVE.value)
                    db_mgr = ctrl_live.engine.db if ctrl_live else None

                    trade_dialog(
                        sp_id=current_sp_name,
                        db_manager=db_mgr,
                        prefill_ticker=current_ticker,
                        prefill_action=action,
                        prefill_price=decision.get(APIKey.TRADE_PRICE.value, 0.0),
                        prefill_shares=decision.get(APIKey.TRADE_SHARES.value, 0)
                    )

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        render_backtest_tab(selected_persona)


if __name__ == "__main__":
    st.set_page_config(page_title="台股量化交易終端", page_icon="📈", layout="wide")
    main()