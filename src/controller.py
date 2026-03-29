import os

from dotenv import load_dotenv

from bt.account import Account
from bt.blackboard import Blackboard
from bt.strategy import build_trading_tree
from bt.strategy_config import PersonaFactory, TradingPersona
from debug import dbg
from ml.engine import QuantAIEngine
from path import PathConfig


class IDSSController:
    """
    IDSS 系統的 UI 介接控制器 (API Endpoint)。
    負責管理引擎生命週期、接收真實帳戶狀態、並回傳結構化的決策報告。
    """
    def __init__(self, ticker: str, api_keys: list[str] = None):
        self.ticker = ticker

        # 如果 UI 沒有傳入 Key，就自動從 .env 拿
        self.api_keys = api_keys if api_keys else self._get_gemini_keys()

        # 實例化底層 AI 引擎
        self.engine = QuantAIEngine(ticker=self.ticker, api_keys=self.api_keys)
        self.is_ready = False

    def _get_gemini_keys(self) -> list[str]:
        """從 .env 讀取並解析逗號分隔的 API Key 池"""
        env_path = PathConfig.GEMINI_KEY
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        keys_str = os.getenv("GEMINI_API_KEYS", "")
        keys_list = [k.strip() for k in keys_str.split(",") if k.strip()]

        if not keys_list:
            dbg.war("⚠️ 未偵測到有效的 GEMINI_API_KEYS，系統將以無 LLM 模式運行。")

        return keys_list

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

    def execute_daily_decision(self,
                               persona: TradingPersona = TradingPersona.MODERATE,
                               current_cash: float = 2000000.0,
                               current_position: int = 0,
                               avg_cost: float = 0.0) -> dict:
        """
        供 UI 點擊「產生今日決策」時觸發。
        可從前端或真實券商 API (如 Shioji) 傳入真實的資金與持倉狀況。

        :return: 結構化的 dict，包含動作、勝率、情緒與最終戰報。
        """
        if not self.is_ready:
            return {"status": "error", "message": "系統尚未載入完成，請先呼叫 load_system()"}

        # 1. AI 引擎預測與新聞分析 (抓取最新特徵)
        prediction_result = self.engine.predict_today()
        if not prediction_result:
            return {"status": "error", "message": "預測失敗，缺乏最新市場資料"}

        # 2. 設定投資性格與行為樹
        strategy_config = PersonaFactory.get_config(persona)
        # 防呆：沒 Key 就強制關閉 LLM 守門員
        strategy_config.enable_llm_oracle = bool(self.api_keys)
        tree = build_trading_tree(strategy_config)

        # 3. 建立黑板與寫入「真實帳戶資料」
        account = Account(cash=current_cash)
        bb = Blackboard(ticker=self.ticker, account=account)

        # 抄寫 AI 預測勝率與情緒
        bb.prob_final = prediction_result["prob_final"]
        bb.prob_xgb = prediction_result["prob_xgb"]
        bb.prob_dl = prediction_result["prob_dl"]
        bb.prob_market_safe = prediction_result["prob_market_safe"]
        bb.sentiment_score = prediction_result["sentiment_score"]
        bb.sentiment_reason = prediction_result["sentiment_reason"]

        # 寫入 UI 傳來的真實部位狀態
        bb.position = current_position
        bb.avg_cost = avg_cost

        # 取得最新收盤價與成交量 (從資料庫拉取最新一筆)
        df_recent = self.engine.db.get_aligned_market_data(self.ticker, []).tail(1)
        if not df_recent.empty:
            bb.current_price = df_recent['Close'].iloc[-1]
            # 實盤中，明日開盤價未發生，這裡用今日收盤價當作虛擬觸價來計算
            bb.executable_price = bb.current_price
            bb.daily_volume = df_recent['Volume'].iloc[-1]
        else:
            bb.current_price = 0.0
            bb.executable_price = 0.0
            bb.daily_volume = 0

        # 4. 執行行為樹決策 (Tick)
        dbg.log("\n🧠 啟動行為樹戰術決策...")
        tree.tick(bb)

        # 5. 打包結構化結果，回傳給 UI 渲染
        return {
            "status": "success",
            "ticker": self.ticker,
            "date": prediction_result["date"],
            "persona": persona.value,
            "decision": {
                "action": bb.action_decision,              # BUY, SELL, HOLD
                "trade_shares": bb.last_trade_shares,      # 欲交易股數
                "trade_price": bb.last_trade_price,        # 觸發價格
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
            "report": bb.gemini_reasoning if bb.gemini_reasoning else f"系統決策為: {bb.action_decision}"
        }
    