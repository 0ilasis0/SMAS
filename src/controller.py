from datetime import datetime

import pandas as pd

from base import KeyManager
from bt.account import Account
from bt.actions import GenerateGeminiReportNode
from bt.blackboard import Blackboard
from bt.const import DecisionAction, LLMCol
from bt.strategy import build_trading_tree
from bt.strategy_config import PersonaFactory, TradingPersona
from data.const import StockCol, TimeUnit, YfInterval
from debug import dbg
from ml.const import FeatureCol, MetaCol
from ml.engine import QuantAIEngine
from ml.model.llm_oracle import TradingMode


class IDSSController:
    """
    IDSS 系統的 UI 介接控制器 (API Endpoint)。
    負責管理引擎生命週期、接收真實帳戶狀態、並回傳結構化的決策報告。
    """
    def __init__(self, ticker: str, oos_days: int):
        self.ticker = ticker
        self.api_keys = KeyManager.get_gemini_keys()
        self.engine: QuantAIEngine = QuantAIEngine(
            ticker=self.ticker,
            oos_days=oos_days,
            api_keys=self.api_keys
        )
        self.is_ready = False

    def load_system(self) -> bool:
        dbg.log(f"[{self.ticker}] 正在初始化 IDSS 決策中樞...")
        if self.engine.load_inference_models():
            self.is_ready = True
            return True
        return False

    def _get_tw_tick_price(self, price: float) -> float:
        """台股專屬：依據股價級距計算合理的跳動單位 (Tick Size)"""
        if price < 10: tick = 0.01
        elif price < 50: tick = 0.05
        elif price < 100: tick = 0.10
        elif price < 500: tick = 0.50
        elif price < 1000: tick = 1.00
        else: tick = 5.00
        return round(round(price / tick) * tick, 2)

    def execute_decision(self,
        current_cash: float,
        current_position: int,
        avg_cost: float,
        persona: TradingPersona,
        mode: TradingMode = TradingMode.SWING,
    ) -> dict:
        if not self.is_ready:
            return {"status": "error", "message": "系統尚未載入完成，請先呼叫 load_system()"}

        # 1. AI 引擎預測與新聞分析
        prediction_result = self.engine.predict_today(mode=mode)
        if not prediction_result:
            return {"status": "error", "message": "預測失敗，缺乏最新市場資料"}

        # 2. 設定投資性格與行為樹
        strategy_config = PersonaFactory.get_config(persona)
        should_run_llm = bool(self.api_keys and mode != TradingMode.DAY_TRADE)
        strategy_config.enable_llm_oracle = False

        tree = build_trading_tree(strategy_config)

        # 3. 建立黑板與寫入「真實帳戶資料」
        account = Account(cash=current_cash)
        bb = Blackboard(ticker=self.ticker, account=account)

        bb.oracle = self.engine.oracle
        bb.prob_final = prediction_result.get(MetaCol.PROB_FINAL, 0.5)
        bb.prob_xgb = prediction_result.get(MetaCol.PROB_XGB, 0.5)
        bb.prob_dl = prediction_result.get(MetaCol.PROB_DL, 0.5)
        bb.prob_market_safe = prediction_result.get(MetaCol.PROB_MARKET_SAFE, 1.0)
        bb.sentiment_score = prediction_result.get(LLMCol.SENTIMENT_SCORE, 5)
        bb.sentiment_reason = prediction_result.get(LLMCol.SENTIMENT_REASON, "無")
        bb.position = current_position
        bb.avg_cost = avg_cost

        current_price = prediction_result.get("current_price", 0.0)
        bb.current_price = current_price
        bb.executable_price = current_price
        bb.daily_volume = prediction_result.get("avg_5d_vol", 0.0)
        bb.bias_20 = prediction_result.get(FeatureCol.BIAS_MONTH, 0.0)
        bb.return_5d = prediction_result.get(FeatureCol.RETURN_5D, 0.0)

        # 4. 執行行為樹決策 (Tick)
        dbg.log(f"\n🧠 啟動行為樹戰術決策 (模式: {mode.value})...")
        tree.tick(bb)

        # 5. 取得決策動作
        action_str = bb.action_decision.value if hasattr(bb.action_decision, 'value') else str(bb.action_decision)
        trade_price = current_price

        # ==========================================
        # 整合美股跳空校正的智慧定價引擎
        # ==========================================
        if action_str in [DecisionAction.BUY, DecisionAction.SELL] and current_price > 0:
            df_recent = self.engine.db.get_daily_data(self.ticker).tail(20)

            # 1. 計算近期真實波動幅度 (ATR)
            atr = current_price * 0.02
            if not df_recent.empty and len(df_recent) > 1:
                prev_close = df_recent[StockCol.CLOSE].shift(1)
                tr1 = df_recent[StockCol.HIGH] - df_recent[StockCol.LOW]
                tr2 = (df_recent[StockCol.HIGH] - prev_close).abs()
                tr3 = (df_recent[StockCol.LOW] - prev_close).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.tail(14).mean()

            # 2. 計算費半 (^SOX) 隔夜跳空連動
            sox_return = 0.0
            df_sox = self.engine.db.get_daily_data('^SOX').tail(2)
            if not df_sox.empty and len(df_sox) >= 2:
                sox_close_last = df_sox[StockCol.CLOSE].iloc[-1]
                sox_close_prev = df_sox[StockCol.CLOSE].iloc[-2]
                sox_return = (sox_close_last - sox_close_prev) / sox_close_prev

            # 判斷是否為電子/科技類股 (簡易防呆：非電子股降低 Beta 影響)
            # 這裡可用更複雜的 mapping，目前先預設 23, 24, 3 字頭為電子股
            is_tech_stock = self.ticker.startswith(('23', '24', '3', '5', '6', '8'))
            beta = 0.4 if is_tech_stock else 0.1

            expected_open_price = current_price * (1 + (sox_return * beta))

            # 3. 獲取指標與台股漲跌停限制
            market_safe = bb.prob_market_safe
            bias_20 = bb.bias_20
            limit_up = self._get_tw_tick_price(current_price * 1.099)
            limit_down = self._get_tw_tick_price(current_price * 0.901)

            # 確保預期開盤價沒有超過漲跌停
            expected_open_price = min(limit_up, max(limit_down, expected_open_price))

            pricing_prefix = f"\n\n💡 **[智慧定價]** "
            if abs(sox_return) > 0.015:
                gap_dir = "大漲" if sox_return > 0 else "重挫"
                pricing_prefix += f"昨夜費半{gap_dir} {sox_return:.1%}，預估今日合理開盤價位移至 {expected_open_price:.2f}。 "

            if should_run_llm:
                dbg.log("\n啟動盤後覆盤：將智慧定價與決策結果交由 Gemini 撰寫戰報...")
                bb.oracle = self.engine.oracle
                report_node = GenerateGeminiReportNode(oracle=self.engine.oracle)

                # 讓 AI 寫報告
                report_node.tick(bb)

            # 4. 決策定價樹
            if action_str == DecisionAction.BUY:
                # 優先權 1：大盤極度危險 (防禦優先)
                if market_safe < 0.35:
                    raw_price = expected_open_price - (1.2 * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"大盤系統性風險高 (安全度 {market_safe:.0%})，即便個股有買進訊號，仍強烈建議掛「大幅拉回之恐慌價」防禦性低接 (建議買價: {trade_price:.2f})。"

                # 優先權 2：如果勝率大於 75%，「或者」勝率65%且新聞情緒極佳(>=8分)，就勇敢追價！
                elif bb.prob_final >= 0.75 or (bb.prob_final >= 0.65 and bb.sentiment_score >= 8):
                    raw_price = expected_open_price - (0.2 * atr)  # 捨棄溢價追擊，改為開盤價微幅低接
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"個股勝率極高 ({bb.prob_final:.0%})。建議掛「預期開盤價之微幅拉回處」積極承接，避免錯失行情 (建議買價: {trade_price:.2f})。"

                # 優先權 3：乖離修復 (跌深摸底)
                elif bias_20 < -0.06:
                    raw_price = expected_open_price - (0.6 * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"具備跌深反彈契機 (月乖離 {bias_20:.1%})，建議掛「合理拉回價」等待盤中洗盤安全摸底 (建議買價: {trade_price:.2f})。"

                # 常規狀態
                else:
                    raw_price = expected_open_price - (0.8 * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"屬常規震盪格局，建議耐心掛「偏低價」等待盤中自然拉回成交 (建議買價: {trade_price:.2f})。"

            elif action_str == DecisionAction.SELL:
                # 優先權 1：絕對弱勢逃命
                if bb.prob_final <= 0.20 or market_safe < 0.3:
                    raw_price = expected_open_price - (0.5 * atr) # 收斂折價幅度，避免賣在最低點
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"空頭動能強或大盤有崩跌風險，建議掛「低於預期開盤價 (折價 0.5ATR)」果斷出脫求現 (建議賣價: {trade_price:.2f})。"

                # 優先權 2：超漲停利
                elif bias_20 > 0.08:
                    raw_price = expected_open_price + (1.2 * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"短線已嚴重超漲 (月乖離 {bias_20:.1%})，建議掛「極端偏高價」等待主力拉抬時停利給追價散戶 (建議賣價: {trade_price:.2f})。"

                # 優先權 3：強勢股逢高調節
                elif bb.prob_final >= 0.7:
                    raw_price = expected_open_price + (0.8 * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"個股依然強勢，建議掛「偏高價」等待盤中衝高時優雅出脫 (建議賣價: {trade_price:.2f})。"

                # 常規狀態
                else:
                    raw_price = expected_open_price + (0.4 * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"趨勢轉弱，建議掛「微幅高於預期開盤」等待反彈時調節 (建議賣價: {trade_price:.2f})。"

            bb.last_trade_price = trade_price

        # 免責聲明
        bb.gemini_reasoning += "\n\n---\n⚠️ **【系統免責聲明】**：本系統之「智慧定價」並未包含除權息預告。若今日為該標的之「除權息交易日」，其實際平盤基準價將大幅低於昨日收盤價，請務必手動取消或重新計算掛單價格，切勿盲目追價！"

        # 6. 打包結構化結果
        final_date = prediction_result.get("date", datetime.now().strftime('%Y-%m-%d'))

        return {
            "status": "success",
            "ticker": self.ticker,
            "date": final_date,
            "mode": mode.value,
            "persona": persona.value if hasattr(persona, 'value') else str(persona),
            "decision": {
                "action": action_str,
                "trade_shares": bb.last_trade_shares,
                "trade_price": trade_price,
            },
            "account_after_trade": {
                "cash_left": bb.cash,
                "position_left": bb.position
            },
            "ai_signals": {
                "final_prob": bb.prob_final,
                "xgb_prob": bb.prob_xgb,
                "dl_prob": bb.prob_dl,
                "market_safe": bb.prob_market_safe
            },
            "sentiment": {
                "score": bb.sentiment_score,
                "reason": bb.sentiment_reason
            },
            "report": bb.gemini_reasoning if bb.gemini_reasoning else f"系統決策為: {action_str}"
        }

    def sync_market_data(self) -> bool:
        dbg.log(f"[{self.ticker}] 接收 UI 指令：啟動例行市場資料同步...")
        try:
            success = self.engine.update_market_data(period=YfInterval.DAILY_MARKET_YEAR, unit=TimeUnit.YEAR)
            if success: dbg.log(f"[{self.ticker}] 資料庫同步完成！最新收盤資料已就緒。")
            else: dbg.error(f"[{self.ticker}] 同步失敗，請檢查網路連線或 API 狀態。")
            return success
        except Exception as e:
            dbg.error(f"[{self.ticker}] 資料同步過程中發生未預期錯誤: {e}")
            return False
