import traceback
from typing import TYPE_CHECKING

import pandas as pd

from const import StatusCol
from controller import IDSSController
from debug import dbg
from ml.const import MetricsCol, ModelAttr, SignalCol
from ml.params import MLDefault
from ui.const import APIKey
from ui.utils import sync_market_data

if TYPE_CHECKING:
    from bt.account import Account

class BatchReportPipeline:
    """
    一鍵批次財報生成管線
    負責迴圈呼叫 IDSSController，統整多檔標的的 AI 決策與財報。
    """
    def __init__(self, account: "Account", sp_id: str, persona):
        self.account = account
        self.sp_id = sp_id
        self.persona = persona

    def run(self, sync_data: bool = False) -> tuple[pd.DataFrame, list]:
        """
        執行批次管線
        :param sync_data: 是否要在產生財報前強制同步最新市場資料
        :return: (df_summary 總結 DataFrame, all_reports 詳細財報 list)
        """
        sp = self.account.get_sub_portfolio(self.sp_id)
        tickers = sp.watch_tickers

        if not tickers:
            dbg.war(f"組合包 {self.sp_id} 內無任何標的，略過批次處理。")
            return pd.DataFrame(), []

        # 決定可動用資金 (各標的獨立平行評估，因此資金額度共用當前水位)
        if sp.use_shared_cash:
            usable_cash = self.account.unallocated_cash
        else:
            usable_cash = sp.allocated_cash

        summary_rows = []
        all_reports = []

        for ticker in tickers:
            dbg.log(f"🚀 [批次管線] 開始處理標的: {ticker}")

            try:
                # 1. 選擇性資料同步
                if sync_data:
                    sync_market_data(ticker)

                # 2. 提取目前真實庫存狀態
                pos_obj = sp.get_position(ticker)
                my_shares = pos_obj.shares
                my_avg_cost = pos_obj.avg_cost

                # 3. 實體化大腦並載入模型
                ctrl = IDSSController(ticker=ticker, oos_days=0)
                if not ctrl.load_system():
                    # 發生狀況無法載入時觸發
                    dbg.war(f"[{ticker}] ⚠️ 缺乏訓練模型權重，寫入未訓練提示。")
                    summary_rows.append({
                        "標的": ticker,
                        "AI 決策": "尚未訓練",
                        "建議股數": "0",
                        "觸發價": "0.00",
                        "目前庫存": f"{my_shares:,}",
                        "綜合勝率": "0.0%"
                    })
                    continue

                # 4. 執行決策引擎
                result = ctrl.execute_decision(
                    available_cash=usable_cash,
                    current_position=my_shares,
                    avg_cost=my_avg_cost,
                    persona=self.persona
                )

                # 5. 確保成功後，抽取結果與 AUC 寫入
                if result and result.get(APIKey.STATUS.value) == StatusCol.SUCCESS:
                    engine = ctrl.engine

                    # 補足 UI 需要的底層模型指標
                    result[MetricsCol.DICT_KEY.value] = {
                        MetricsCol.XGB_AUC: getattr(engine.xgb_model, ModelAttr.VAL_AUC, MLDefault.FALLBACK_AUC),
                        MetricsCol.DL_AUC: getattr(engine.dl_model, ModelAttr.VAL_AUC, MLDefault.FALLBACK_AUC),
                        MetricsCol.MARKET_AUC: getattr(engine.market_model, ModelAttr.VAL_AUC, MLDefault.FALLBACK_AUC),
                        MetricsCol.MARKET_THRESH: getattr(engine.market_model, ModelAttr.DYNAMIC_THRESH, MLDefault.FALLBACK_THRESH)
                    }

                    decision = result.get(APIKey.DECISION.value, {})
                    action = decision.get(APIKey.ACTION.value, "HOLD")
                    shares = decision.get(APIKey.TRADE_SHARES.value, 0)
                    price = decision.get(APIKey.TRADE_PRICE.value, 0.0)
                    ai_sigs = result.get(APIKey.AI_SIGNALS.value, {})
                    prob_final = ai_sigs.get(SignalCol.PROB_FINAL.value, 0.0)

                    # 整理到總表 Row
                    summary_rows.append({
                        "標的": ticker,
                        "AI 決策": action,
                        "建議股數": f"{shares:,}",
                        "觸發價": f"{price:,.2f}",
                        "目前庫存": f"{my_shares:,}",
                        "綜合勝率": f"{prob_final:.1%}"
                    })

                    # 將完整結構保留供 UI 的 expander 展開閱讀
                    all_reports.append({
                        "ticker": ticker,
                        "action": action,
                        "raw_result": result
                    })
                    dbg.log(f"[{ticker}] ✅ 批次處理成功！決策: {action}")
                else:
                    err_msg = result.get(APIKey.MESSAGE.value) if result else "未知錯誤"
                    dbg.error(f"[{ticker}] ❌ 決策執行失敗: {err_msg}")

            except Exception as e:
                dbg.error(f"[{ticker}] ❌ 批次管線發生崩潰: {e}\n{traceback.format_exc()}")
                continue

        # 將 dict 轉換成 DataFrame，準備交給 Streamlit 渲染
        df_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
        return df_summary, all_reports