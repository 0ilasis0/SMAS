from enum import StrEnum

import streamlit as st

from bt.backtest import BacktestEngine
from bt.strategy_config import PersonaFactory
from controller import IDSSController
from path import PathConfig
from ui.const import SessionKey
from ui.params import AccountLimit, BacktestParams


class AssetSource(StrEnum):
    GLOBAL = "global"
    CUSTOM = "custom"

class RiskMode(StrEnum):
    REALISTIC = "realistic"
    SINGLE_STOCK = "single_stock"

DISPLAY_ASSET_SOURCE = {
    AssetSource.GLOBAL: f"全域預設資金 ({AccountLimit.DEFAULT_GLOBAL:,} 元)",
    AssetSource.CUSTOM: "自訂單檔投入預算"
}

DISPLAY_RISK_MODE = {
    RiskMode.REALISTIC: "實戰模擬 (動態水位風控)",
    RiskMode.SINGLE_STOCK: "個股模擬 (純粹訊號極限)"
}

def render_backtest_tab(selected_persona):
    """渲染歷史回測模擬介面 (支援動態載入 OOS=240 大腦)"""
    st.markdown("### ⚙️ 回測參數設定")

    col1, col2 = st.columns(2)

    with col1:
        test_days = st.slider(
            "📅 模擬時間線 (交易日)",
            min_value=BacktestParams.MIN_DAYS,
            max_value=BacktestParams.MAX_DAYS,
            value=BacktestParams.DEFAULT_DAYS,
            step=BacktestParams.STEP_DAYS
        )

    with col2:
        asset_option = st.radio(
            "💰 模擬資金來源",
            options=[AssetSource.GLOBAL, AssetSource.CUSTOM],
            format_func=lambda x: DISPLAY_ASSET_SOURCE.get(x, x),
            help="建議為該股票建立一個獨立的虛擬子帳戶，避免閒置現金拖累回測績效。"
        )

        if asset_option == AssetSource.CUSTOM:
            sim_cash = st.number_input(
                "請輸入投入預算 (NTD)",
                min_value=AccountLimit.MIN_MONEY,
                max_value=AccountLimit.MAX_MONEY,
                value=AccountLimit.DEFAULT_SINGLE,
                step=AccountLimit.STEP_MONEY,
                format="%d"
            )
        else:
            sim_cash = float(AccountLimit.DEFAULT_GLOBAL)

    st.markdown("#### 🛡️ 風險控制系統設定")
    risk_mode_label = st.radio(
        "行為樹風控層級",
        options=[RiskMode.REALISTIC, RiskMode.SINGLE_STOCK],
        format_func=lambda x: DISPLAY_RISK_MODE.get(x, x),
        horizontal=True,
        help="【實戰模擬】會根據庫存比例動態調整買賣門檻。\n【個股模擬】則可能產生極端操作。"
    )

    is_pure_signal_test = (risk_mode_label == RiskMode.SINGLE_STOCK)

    st.markdown("---")

    if st.button("🚀 啟動 IDSS 歷史回測", type="primary", use_container_width=True):
        with st.spinner(f"正在擷取近 {test_days} 天 AI 預測勝率，並進行沙盤推演..."):

            current_ticker = st.session_state.get(SessionKey.CURRENT_TICKER.value)

            # 動態喚醒「回測專用大腦」
            if st.session_state.get(SessionKey.CTRL_BT.value) is None:
                st.toast(f"首次啟動回測，正在載入 OOS={BacktestParams.MAX_DAYS} 的盲測模型...", icon="⏳")

                ctrl_bt = IDSSController(ticker=current_ticker, oos_days=BacktestParams.MAX_DAYS)
                if ctrl_bt.load_system():
                    st.session_state[SessionKey.CTRL_BT.value] = ctrl_bt
                else:
                    st.error(f"❌ 找不到 OOS={BacktestParams.MAX_DAYS} 的回測模型，請先在後台完成訓練！")
                    st.stop()

            ctrl_bt = st.session_state.get(SessionKey.CTRL_BT.value)
            df_bt = ctrl_bt.engine.generate_backtest_data()

            if not df_bt.empty:
                df_test = df_bt.tail(test_days)
                strategy_config = PersonaFactory.get_config(selected_persona)

                # 盲測時關閉 LLM (避免觸發大量新聞 API 消耗與幻覺)
                strategy_config.enable_llm_oracle = False

                bt_engine = BacktestEngine(initial_cash=sim_cash, ticker=current_ticker, strategy=strategy_config)

                # 將 UI 的風控模式選擇，注入到引擎的黑板中
                bt_engine.bb.is_backtest = is_pure_signal_test

                stats = bt_engine.run(df_test)

                if stats:
                    st.success(f"✅ {test_days} 天歷史回測推演完成！")
                    st.markdown("### 🏆 IDSS 策略績效總覽")

                    m1, m2, m3, m4 = st.columns(4)
                    profit_color = "normal" if stats['total_return'] > 0 else "inverse"
                    m1.metric("🏦 最終總淨值", f"${stats['final_equity']:,.0f}", f"初始: ${stats['initial_cash']:,.0f}", delta_color="off")
                    m2.metric("📈 區間總報酬率", f"{stats['total_return']:.2%}", delta=f"{stats['total_return']:.2%}", delta_color=profit_color)
                    m3.metric("🚀 年化報酬 (CAGR)", f"{stats['cagr']:.2%}")
                    m4.metric("📉 最大回撤 (MDD)", f"{stats['mdd']:.2%}", delta="風險評估", delta_color="off")

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("⚖️ 夏普值 (Sharpe)", f"{stats['sharpe']:.2f}")
                    m6.metric("🛒 AI 買進次數", f"{stats['buy_count']} 次")
                    m7.metric("💸 AI 賣出次數", f"{stats['sell_count']} 次")
                    m8.metric("📊 總交易頻率", f"{stats['buy_count'] + stats['sell_count']} 次")

                    st.markdown("---")

                    chart_path = PathConfig.get_chart_report_path(current_ticker)
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)
                    else:
                        st.error("❌ 找不到回測圖表，請確認 BacktestEngine 有成功儲存圖片。")
            else:
                st.error("❌ 無法生成回測資料，請確認模型是否已訓練完畢。")