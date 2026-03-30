import streamlit as st

# 依照你實際的檔案路徑引入 TradingPersona
from bt.strategy_config import TradingPersona
# 引入我們剛剛封裝好的 Controller 與 性格設定
from controller import IDSSController

# 1. 頁面基礎設定 (必須放在最前面)
st.set_page_config(page_title="IDSS 量化決策中樞", page_icon="📈", layout="wide")

# 2. 初始化 Session State (確保模型只載入一次，不會每次點擊都重跑)
if "controller" not in st.session_state:
    st.session_state.controller = None
if "is_loaded" not in st.session_state:
    st.session_state.is_loaded = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# 側邊欄 (Sidebar) - 使用者輸入區
# ==========================================
with st.sidebar:
    st.header("⚙️ 系統核心設定")
    ticker = st.text_input("股票代號", value="2337.TW")

    # 載入模型按鈕
    if st.button("🔄 載入/切換 AI 模型"):
        with st.spinner(f"正在載入 {ticker} 的深度學習權重與 Scaler..."):
            st.session_state.controller = IDSSController(ticker=ticker)
            success = st.session_state.controller.load_system()
            if success:
                st.session_state.is_loaded = True
                st.success("✅ 模型載入完成！")
            else:
                st.session_state.is_loaded = False
                st.error("❌ 載入失敗，請檢查模型檔案是否存在。")

    st.divider()

    st.header("👤 帳戶與性格模擬")
    # 將中文選項映射到 Enum
    persona_map = {
        "激進型 (Aggressive)": TradingPersona.AGGRESSIVE,
        "穩健型 (Moderate)": TradingPersona.MODERATE,
        "保守型 (Conservative)": TradingPersona.CONSERVATIVE
    }
    selected_persona_str = st.selectbox("投資性格", options=list(persona_map.keys()), index=1)

    current_cash = st.number_input("可用資金 (NTD)", value=2000000, step=100000)
    current_pos = st.number_input("目前持股 (股)", value=0, step=1000)
    avg_cost = st.number_input("平均成本 (元)", value=0.0, step=1.0)

    # 產生決策的主按鈕 (使用 primary 顏色醒目提示)
    generate_btn = st.button("🚀 產生今日 AI 決策", type="primary", use_container_width=True)


# ==========================================
# 主畫面 (Main Content) - 戰情儀表板
# ==========================================
st.title("📈 IDSS 智能量化交易戰情室")

# 當使用者點擊「產生決策」時的執行邏輯
if generate_btn:
    if not st.session_state.is_loaded or st.session_state.controller is None:
        st.warning("⚠️ 系統尚未就緒，請先在左側輸入代號並點擊「載入/切換 AI 模型」！")
    else:
        with st.spinner("🧠 雙腦運算中，並呼叫 Gemini 讀取最新新聞情緒..."):
            persona_enum = persona_map[selected_persona_str]
            # 呼叫 Controller 拿取字典結果
            result = st.session_state.controller.execute_daily_decision(
                persona=persona_enum,
                current_cash=current_cash,
                current_position=current_pos,
                avg_cost=avg_cost
            )
            # 將結果存入暫存，畫面刷新時才不會消失
            st.session_state.last_result = result

# --- 渲染決策結果 ---
if st.session_state.last_result:
    res = st.session_state.last_result

    if res.get("status") == "error":
        st.error(res.get("message"))
    else:
        st.subheader(f"📅 決策日期：{res['date']} | 標的：{res['ticker']} | 性格：{res['persona']}")

        # 1. 核心決策大字報
        action = res["decision"]["action"]
        # 根據動作決定顏色
        color = "#00cc66" if action == "BUY" else "#ff4d4d" if action == "SELL" else "#8c8c8c"
        st.markdown(f"<h1 style='text-align: center; color: {color};'>最終戰術動作：{action}</h1>", unsafe_allow_html=True)

        # 使用 Streamlit 內建的 metric 排列三個關鍵數字
        col1, col2, col3 = st.columns(3)
        col1.metric("預計交易股數", f"{res['decision']['trade_shares']:,}")
        col2.metric("觸發價格", f"${res['decision']['trade_price']:.2f}")
        col3.metric("交易後剩餘資金", f"${res['account_after_trade']['cash_left']:,.0f}")

        st.divider()

        # 2. AI 訊號儀表板
        st.subheader("🧠 AI 多模型大腦解析")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總指揮勝率 (Meta)", f"{res['ai_signals']['final_prob']:.2%}")
        c2.metric("DL 右腦 (K線型態)", f"{res['ai_signals']['dl_prob']:.2%}")
        c3.metric("XGB 左腦 (技術指標)", f"{res['ai_signals']['xgb_prob']:.2%}")
        c4.metric("大盤安全度 (TWII)", f"{res['ai_signals']['market_safe']:.2%}")

        st.divider()

        # 3. Gemini 神諭機戰報區
        st.subheader("📰 Gemini 神諭機戰報")
        # 顯示分數與理由
        st.info(f"**情緒分數：{res['sentiment']['score']} / 10** ｜ 判讀理由：{res['sentiment']['reason']}")
        # 顯示最終法人報告
        st.success(res['report'])
