from datetime import datetime

import pandas as pd

from base import KeyManager, MathTool
from bt.account import Account, Position, SubPortfolio
from bt.actions import GenerateGeminiReportNode
from bt.blackboard import Blackboard
from bt.const import AccountCol, TradeDecision
from bt.strategy import build_trading_tree
from bt.strategy_config import PersonaFactory, TradingPersona
from const import GlobalParams
from data.const import StockCol
from data.fetcher import Fetcher
from data.updater import DataUpdater
from debug import dbg
from ml.const import FeatureCol, OracleCol, QuoteCol, SignalCol, TradingMode
from ml.engine import QuantAIEngine
from ui.const import APIKey


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
        available_cash: float,
        current_position: int,
        avg_cost: float,
        persona: TradingPersona,
    ) -> dict:
        if not self.is_ready:
            return {"status": "error", "message": "系統尚未載入完成，請先呼叫 load_system()"}

        # 1. AI 引擎預測與新聞分析
        prediction_result = self.engine.predict_today(mode=TradingMode.SWING, is_t_minus_1_sim=GlobalParams.IS_T_MINUS_1_SIM)
        if not prediction_result:
            return {"status": "error", "message": "預測失敗，缺乏最新市場資料"}

        # 取得系統當前判定的目標日期
        current_date = prediction_result.get(QuoteCol.DATE.value, datetime.now().strftime('%Y-%m-%d'))

        # 除權息假跌破護盾 (在放上黑板之前執行)
        div_info = self.engine.db.get_upcoming_dividend(self.ticker, current_date)
        dividend_warning = ""
        if div_info and div_info.get('ex_date') == current_date and current_position > 0:
            cash_div = div_info.get('cash_dividend', 0.0)
            avg_cost = max(0.0, avg_cost - cash_div)
            dividend_warning = f"\n\n💰 **[除權息校正]** 今日為 {self.ticker} 除息日 (配發現金 {cash_div} 元)。為防止系統觸發假跌破停損，您的持倉成本已自動從帳面上扣除配息，下調至 {avg_cost:.2f} 元。"
            dbg.log(f"[{self.ticker}] 觸發除權息護盾！成本下修 {cash_div} 元。")

        # 2. 設定投資性格與行為樹
        strategy_config = PersonaFactory.get_config(persona)
        should_run_llm = bool(self.api_keys)
        strategy_config.enable_llm_oracle = False
        tree = build_trading_tree(strategy_config)

        # 3. 建立黑板與寫入「真實帳戶資料」(虛擬推演用)
        account = Account(total_cash=available_cash)
        real_display_price = prediction_result.get(QuoteCol.REAL_LATEST_PRICE.value, avg_cost)

        account.sub_portfolios[AccountCol.DUMMY_SP.value] = SubPortfolio(name=AccountCol.DUMMY_SP.value)
        account.sub_portfolios[AccountCol.DUMMY_SP.value].positions[self.ticker] = Position(
            shares=current_position,
            avg_cost=avg_cost,
            current_price=real_display_price
        )

        bb = Blackboard(ticker=self.ticker, account=account)
        bb.is_backtest = False

        bb.prob_final = prediction_result.get(SignalCol.PROB_FINAL.value, GlobalParams.DEFAULT_ERROR)
        bb.prob_xgb = prediction_result.get(SignalCol.PROB_XGB.value, GlobalParams.DEFAULT_ERROR)
        bb.prob_dl = prediction_result.get(SignalCol.PROB_DL.value, GlobalParams.DEFAULT_ERROR)
        bb.prob_market_safe = prediction_result.get(SignalCol.PROB_MARKET_SAFE.value, GlobalParams.DEFAULT_ERROR)
        bb.sentiment_score = prediction_result.get(OracleCol.SCORE.value, 5)
        bb.sentiment_reason = prediction_result.get(OracleCol.REASON.value, "無")

        bb.position = current_position
        bb.avg_cost = avg_cost
        current_price = prediction_result.get(QuoteCol.CURRENT_PRICE.value, 0.0)
        bb.current_price = current_price
        bb.executable_price = current_price
        bb.daily_volume = prediction_result.get(QuoteCol.AVG_5D_VOL.value, 0.0)
        bb.bias_20 = prediction_result.get(FeatureCol.BIAS_MONTH.value, 0.0)
        bb.return_5d = prediction_result.get(FeatureCol.RETURN_5D.value, 0.0)

        if any(p < 0 for p in [bb.prob_final, bb.prob_xgb, bb.prob_dl, bb.prob_market_safe]):
            dbg.error(f"🚨 [致命錯誤] 檢測到 AI 引擎輸出異常機率值 ({GlobalParams.DEFAULT_ERROR})，神經網路可能已崩潰或特徵缺失！")
            return {
                "status": "error",
                "message": "AI 引擎推論失敗 (機率異常)，為保護資金安全，系統已強制熔斷並鎖死交易功能。"
            }

        # 4. 執行行為樹決策 (Tick)
        dbg.log(f"\n🧠 啟動行為樹戰術決策 (波段模式)...")
        tree.tick(bb)

        account.sub_portfolios[AccountCol.DUMMY_SP.value].positions[self.ticker].shares = bb.position
        account.sub_portfolios[AccountCol.DUMMY_SP.value].positions[self.ticker].avg_cost = bb.avg_cost

        bb.gemini_reasoning = dividend_warning # 帶入剛才的除息警告 (如果有)
        action_str = bb.action_decision.value if hasattr(bb.action_decision, 'value') else str(bb.action_decision)
        trade_price = 0.0

        # 法說會避險
        days_to_earnings = self.engine.db.get_days_to_next_earnings(self.ticker, current_date)

        if action_str == TradeDecision.BUY.value and days_to_earnings is not None and 0 <= days_to_earnings <= strategy_config.earnings_shield_days:
            action_str = TradeDecision.HOLD.value
            bb.action_decision = TradeDecision.HOLD
            bb.gemini_reasoning += f"\n\n🚨 **[事件避險]** 距離本檔股票的法說會/財報公佈僅剩 **{days_to_earnings} 天**。為防範財報地雷引發的無預警跳空大跌，系統已強制撤銷買進決策，改為【觀望】。君子不立危牆之下。"
            dbg.log(f"[{self.ticker}] 觸發法說會避險！強制取消 BUY 訊號。")

        if action_str in [TradeDecision.BUY.value, TradeDecision.SELL.value] and current_price > 0:
            df_recent = self.engine.db.get_daily_data(self.ticker).tail(20)

            atr = current_price * 0.02
            if not df_recent.empty and len(df_recent) > 1:
                prev_close = df_recent[StockCol.CLOSE.value].shift(1)
                tr1 = df_recent[StockCol.HIGH.value] - df_recent[StockCol.LOW.value]
                tr2 = (df_recent[StockCol.HIGH.value] - prev_close).abs()
                tr3 = (df_recent[StockCol.LOW.value] - prev_close).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.tail(14).mean()

            sox_return = 0.0
            df_sox = self.engine.db.get_daily_data('^SOX').tail(2)
            if not df_sox.empty and len(df_sox) >= 2:
                sox_close_last = df_sox[StockCol.CLOSE.value].iloc[-1]
                sox_close_prev = df_sox[StockCol.CLOSE.value].iloc[-2]
                sox_return = (sox_close_last - sox_close_prev) / sox_close_prev

            is_tech_stock = self.ticker.startswith(('23', '24', '3', '5', '6', '8'))
            beta = strategy_config.beta_tech if is_tech_stock else strategy_config.beta_non_tech
            raw_expected_open = current_price * (1 + (sox_return * beta))

            market_safe = bb.prob_market_safe
            bias_20 = bb.bias_20

            limit_up = self._get_tw_tick_price(current_price * strategy_config.tw_limit_up_ratio)
            limit_down = self._get_tw_tick_price(current_price * strategy_config.tw_limit_down_ratio)

            raw_expected_open = MathTool.clamp(raw_expected_open, limit_down, limit_up)
            expected_open_price = self._get_tw_tick_price(raw_expected_open)

            pricing_prefix = f"\n\n💡 **[智慧定價]** "
            if abs(sox_return) > strategy_config.sox_surge_threshold:
                gap_dir = "大漲" if sox_return > 0 else "重挫"
                pricing_prefix += f"昨夜費半{gap_dir} {sox_return:.1%}，預期今日開盤價位移至 {expected_open_price:.2f}。 "

            if action_str == TradeDecision.BUY.value:
                if market_safe < strategy_config.market_danger_threshold:
                    raw_price = expected_open_price - (strategy_config.buy_panic_discount_atr * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"大盤系統性風險高 (安全度 {market_safe:.0%})，即便個股有買進訊號，仍強烈建議掛「大幅拉回之恐慌價」防禦性低接 (建議買價: {trade_price:.2f})。"

                elif bb.prob_final >= strategy_config.pricing_buy_extreme_prob or (bb.prob_final >= strategy_config.pricing_buy_strong_prob and bb.sentiment_score >= strategy_config.pricing_buy_sentiment_min):
                    raw_price = expected_open_price - (strategy_config.buy_strong_discount_atr * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"個股勝率極高 ({bb.prob_final:.0%})。建議掛「預期開盤價之微幅拉回處」積極承接，避免錯失行情 (建議買價: {trade_price:.2f})。"

                elif bias_20 < strategy_config.buy_rebound_bias:
                    raw_price = expected_open_price - (strategy_config.buy_rebound_discount_atr * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"具備跌深反彈契機 (月乖離 {bias_20:.1%})，建議掛「合理拉回價」等待盤中洗盤安全摸底 (建議買價: {trade_price:.2f})。"
                else:
                    raw_price = expected_open_price - (strategy_config.buy_normal_discount_atr * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"屬常規震盪格局，建議耐心掛「偏低價」等待盤中自然拉回成交 (建議買價: {trade_price:.2f})。"

            elif action_str == TradeDecision.SELL.value:

                if bb.prob_final <= strategy_config.pricing_sell_extreme_prob or market_safe < strategy_config.hold_weak_threshold:
                    raw_price = expected_open_price - (strategy_config.sell_panic_discount_atr * atr)
                    trade_price = max(limit_down, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"空頭動能強或大盤有崩跌風險，建議掛「低於預期開盤價 (折價 {strategy_config.sell_panic_discount_atr}ATR)」果斷出脫求現 (建議賣價: {trade_price:.2f})。"

                elif bias_20 > strategy_config.sell_overheated_bias:
                    raw_price = expected_open_price + (strategy_config.sell_overheated_premium_atr * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"短線已嚴重超漲 (月乖離 {bias_20:.1%})，建議掛「偏高價」等待主力拉抬時停利給追價散戶 (建議賣價: {trade_price:.2f})。"

                elif bb.prob_final >= strategy_config.pricing_sell_strong_prob:
                    raw_price = expected_open_price + (strategy_config.sell_strong_premium_atr * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"個股依然強勢，建議掛「偏高價」等待盤中衝高時優雅出脫 (建議賣價: {trade_price:.2f})。"
                else:
                    raw_price = expected_open_price + (strategy_config.sell_normal_premium_atr * atr)
                    trade_price = min(limit_up, self._get_tw_tick_price(raw_price))
                    bb.gemini_reasoning += pricing_prefix + f"趨勢轉弱，建議掛「微幅高於預期開盤」等待反彈時調節 (建議賣價: {trade_price:.2f})。"

            bb.last_trade_price = trade_price

        elif action_str == TradeDecision.HOLD.value:
            hold_reason = f"\n\n\n**[觀望判定]** "

            if days_to_earnings is not None and 0 <= days_to_earnings <= strategy_config.earnings_shield_days and bb.action_decision == TradeDecision.HOLD:
                 hold_reason += f"系統已啟動【法說會避險機制】強制攔截交易。即便原模型算定勝率極高，仍強制退回空手觀望，防範財報地雷，確保資金絕對安全。"
            elif bb.prob_market_safe < strategy_config.hold_danger_threshold:
                if current_position > 0: hold_reason += f"大盤系統風險高 (安全度 {bb.prob_market_safe:.0%})，但訊號未達停損閥值。強烈建議嚴格控管既有部位風險，跌破支撐果斷離場。"
                else: hold_reason += f"大盤系統風險高 (安全度 {bb.prob_market_safe:.0%})，目前空手，系統強制壓抑交易衝動，以資金避險優先。"
            elif bb.prob_final <= strategy_config.hold_weak_threshold and current_position == 0:
                hold_reason += f"綜合技術面偏空 (勝率 {bb.prob_final:.0%})，具備下跌風險。因帳上無庫存，維持空手觀望。"
            elif bb.prob_final < strategy_config.hold_neutral_threshold and current_position == 0:
                hold_reason += f"個股動能不足 (勝率 {bb.prob_final:.0%})，處於盤整期，不具建倉優勢，建議保留現金實力。"
            elif bb.prob_final >= strategy_config.hold_wait_threshold and current_position > 0:
                hold_reason += f"個股勝率適中 ({bb.prob_final:.0%})，趨勢未明，建議既有部位續抱觀察，靜待表態。"
            else:
                hold_reason += f"動能訊號未達閥值 (勝率 {bb.prob_final:.0%})，建議維持現狀，避免耗損交易成本。"

            bb.gemini_reasoning += hold_reason
            bb.last_trade_price = 0.0

        atr_ratio = prediction_result.get(FeatureCol.ATR_RATIO.value, 0.0)
        trend_strength = prediction_result.get(FeatureCol.TREND_STRENGTH.value, 0.0)

        if bb.prob_final < strategy_config.wash_risk_win_rate and atr_ratio > strategy_config.wash_risk_atr_ratio:
            warning_msg = (
                f"\n\n🚨 **[總裁特級指令]** 目前個股真實波幅 (ATR Ratio) 高達 {atr_ratio:.2%}，"
                f"且趨勢強度呈現不穩 ({trend_strength:.2f})，AI 綜合勝率偏低 ({bb.prob_final:.0%})。"
                f"請你在戰報的開頭，以強烈的語氣加上一段『⚠️ 總裁警告』，明確告訴使用者："
                f"『量化模型偵測到該股目前處於高波動的洗盤階段。由於未來極易觸發停損機制（接刀子風險），AI 已主動下修買進評分。強烈建議目前以保護資金為主，空手觀望，等待波動率收斂且底打好後再行評估。』"
            )
            bb.gemini_reasoning += warning_msg

        if should_run_llm:
            dbg.log("\n啟動盤後覆盤：將智慧定價與決策結果交由 Gemini 撰寫戰報...")
            bb.oracle = self.engine.oracle
            report_node = GenerateGeminiReportNode(oracle=self.engine.oracle)
            report_node.tick(bb)

        final_date = prediction_result.get(QuoteCol.DATE.value, datetime.now().strftime('%Y-%m-%d'))

        return {
            APIKey.STATUS.value: "success",
            QuoteCol.TICKER.value: self.ticker,
            QuoteCol.DATE.value: final_date,
            APIKey.MODE.value: TradingMode.SWING.value,
            APIKey.PERSONA.value: persona.value if hasattr(persona, 'value') else str(persona),

            APIKey.DECISION.value: {
                APIKey.ACTION.value: action_str,
                APIKey.TRADE_SHARES.value: bb.last_trade_shares,
                APIKey.TRADE_PRICE.value: trade_price,
            },

            APIKey.ACCOUNT.value: {
                APIKey.CASH_LEFT.value: account.total_cash,
                APIKey.POSITION_LEFT.value: bb.position,
                APIKey.TOTAL_EQUITY: account.total_equity
            },

            APIKey.AI_SIGNALS.value: {
                SignalCol.PROB_FINAL.value: bb.prob_final,
                SignalCol.PROB_XGB.value: bb.prob_xgb,
                SignalCol.PROB_DL.value: bb.prob_dl,
                SignalCol.PROB_MARKET_SAFE.value: bb.prob_market_safe
            },

            APIKey.SENTIMENT.value: {
                OracleCol.SCORE.value: bb.sentiment_score,
                OracleCol.REASON.value: bb.sentiment_reason
            },

            APIKey.REPORT.value: bb.gemini_reasoning if bb.gemini_reasoning else f"系統決策為: {action_str}"
        }

    def sync_market_data(self) -> bool:
        dbg.log(f"[{self.ticker}] 接收 UI 指令：啟動例行市場資料同步...")
        try:
            updater = DataUpdater(self.engine.db, Fetcher())
            success = updater.update_market_data(ticker=self.ticker, force_sync=True)

            if success: dbg.log(f"[{self.ticker}] 資料庫同步完成！最新收盤資料已就緒。")
            else: dbg.error(f"[{self.ticker}] 同步失敗，請檢查網路連線或 API 狀態。")
            return success
        except Exception as e:
            dbg.error(f"[{self.ticker}] 資料同步過程中發生未預期錯誤: {e}")
            return False