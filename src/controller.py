from datetime import datetime

from base import KeyManager
from bt.account import Account
from bt.blackboard import Blackboard
from bt.const import LLMCol
from bt.strategy import build_trading_tree
from bt.strategy_config import PersonaFactory, TradingPersona
from data.const import TimeUnit, YfInterval
from debug import dbg
from ml.const import MetaCol
from ml.engine import QuantAIEngine
from ml.model.llm_oracle import TradingMode


class IDSSController:
    """
    IDSS 系統的 UI 介接控制器 (API Endpoint)。
    負責管理引擎生命週期、接收真實帳戶狀態、並回傳結構化的決策報告。
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.api_keys = KeyManager.get_gemini_keys()
        # 實例化底層 AI 引擎
        self.engine: QuantAIEngine = QuantAIEngine(ticker=self.ticker, api_keys=self.api_keys)
        self.is_ready = False

    def load_system(self) -> bool:
        """
        供 UI 啟動時觸發 (可用於顯示 Loading 進度條)。
        載入所有 AI 權重與 Scaler。
        """
        dbg.log(f"[{self.ticker}] 正在初始化 IDSS 決策中樞...")
        if self.engine.load_inference_models():
            self.is_ready = True
            return True
        return False

    def execute_decision(self,
        current_cash: float,
        current_position: int,
        avg_cost: float,
        persona: TradingPersona,
        mode: TradingMode = TradingMode.SWING,
    ) -> dict:
        """
        供 UI 點擊「產生決策」時觸發。
        支援當沖(DAY_TRADE) 與 波段(SWING) 模式切換。
        """
        if not self.is_ready:
            return {"status": "error", "message": "系統尚未載入完成，請先呼叫 load_system()"}

        # 1. AI 引擎預測與新聞分析
        prediction_result = self.engine.predict_today(mode=mode)
        if not prediction_result:
            return {"status": "error", "message": "預測失敗，缺乏最新市場資料"}

        # 2. 設定投資性格與行為樹
        strategy_config = PersonaFactory.get_config(persona)

        # 防呆：沒 Key 或處於當沖模式時，皆關閉 LLM 守門員
        if not self.api_keys or mode == TradingMode.DAY_TRADE:
            strategy_config.enable_llm_oracle = False
        else:
            strategy_config.enable_llm_oracle = True

        tree = build_trading_tree(strategy_config)

        # 3. 建立黑板與寫入「真實帳戶資料」
        account = Account(cash=current_cash)
        bb = Blackboard(ticker=self.ticker, account=account)

        # 將遺漏的特徵完整抄寫到黑板
        bb.prob_final = prediction_result.get(MetaCol.PROB_FINAL, 0.5)
        bb.prob_xgb = prediction_result.get(MetaCol.PROB_XGB, 0.5)
        bb.prob_dl = prediction_result.get(MetaCol.PROB_DL, 0.5)
        bb.prob_market_safe = prediction_result.get(MetaCol.PROB_MARKET_SAFE, 1.0) # 補上這行

        bb.sentiment_score = prediction_result.get(LLMCol.SENTIMENT_SCORE, 5)           # 補上這行
        bb.sentiment_reason = prediction_result.get(LLMCol.SENTIMENT_REASON, "無")

        # 寫入 UI 傳來的真實部位狀態
        bb.position = current_position
        bb.avg_cost = avg_cost

        bb.current_price = prediction_result.get("current_price", 0.0)
        bb.executable_price = bb.current_price # 實盤觸價妥協
        bb.daily_volume = prediction_result.get("avg_5d_vol", 0.0)

        # 4. 執行行為樹決策 (Tick)
        dbg.log(f"\n🧠 啟動行為樹戰術決策 (模式: {mode.value})...")
        tree.tick(bb)

        # 5. 打包結構化結果
        action_str = bb.action_decision.value if hasattr(bb.action_decision, 'value') else str(bb.action_decision)

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
                "trade_price": bb.last_trade_price,
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
        """
        供 UI 點擊「🔄 同步並載入最新資料」時觸發的唯一窗口。
        職責：只負責輕量級的「例行更新」（抓取近期個股與大盤資料）。
        *若市場發生股票分割，新抓入的資料會與舊資料產生斷崖，
         隨後預測時將由底層的 Watchdog 自動偵測並觸發深度清洗。*
        """
        dbg.log(f"[{self.ticker}] 接收 UI 指令：啟動例行市場資料同步...")

        try:
            success = self.engine.update_market_data(period=YfInterval.DAILY_MARKET_YEAR, unit=TimeUnit.YEAR)

            if success:
                dbg.log(f"[{self.ticker}] 資料庫同步完成！最新收盤資料已就緒。")
            else:
                dbg.error(f"[{self.ticker}] 同步失敗，請檢查網路連線或 API 狀態。")

            return success

        except Exception as e:
            dbg.error(f"[{self.ticker}] 資料同步過程中發生未預期錯誤: {e}")
            return False