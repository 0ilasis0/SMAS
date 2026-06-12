import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from base import KeyManager
from const import StatusCol
from debug import dbg


class HybridEventFetcher:
    """
    事件抓取備援引擎 (FinMind 優先 -> yfinance 備援)。
    內建斷路器機制，若 FinMind 連續失敗，當日將永久切換至 yfinance。
    """
    # 紀錄 FinMind 是否已經掛掉
    _finmind_circuit_breaker_tripped = False
    # 設定基礎的網路重試次數
    MAX_RETRIES = 3

    def __init__(self):
        self.finmind_keys_pool = KeyManager.get_finmind_keys()
        self.current_key_index = 0
        self.finmind_url = "https://api.finmindtrade.com/api/v4/data"

    def fetch_upcoming_events(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """同時抓取除權息與法說會，回傳 (df_div, df_earn)"""

        # 如果斷路器已經觸發，直接跳過 FinMind
        if not HybridEventFetcher._finmind_circuit_breaker_tripped:
            df_div, df_earn = self._fetch_from_finmind(ticker)
            if not df_div.empty or not df_earn.empty:
                return df_div, df_earn

            # 如果 FinMind 回傳空值或報錯，觸發斷路器
            dbg.war("⚠️ FinMind 抓取失敗，觸發斷路器！今日後續更新將全面改用 yfinance。")
            HybridEventFetcher._finmind_circuit_breaker_tripped = True

        # 使用備援機制
        return self._fetch_from_yfinance(ticker)

    def _rotate_key(self) -> bool:
        """切換到下一把金鑰。回傳 True 代表切換成功；回傳 False 代表金鑰用盡。"""
        if not self.finmind_keys_pool:
            return False

        self.current_key_index += 1

        if self.current_key_index >= len(self.finmind_keys_pool):
            dbg.error("❌ 所有 FinMind 金鑰皆已耗盡或遭到限制！即將觸發斷路器。")
            return False

        dbg.log(f"🔄 偵測到 FinMind Token 受限，已切換至備用金鑰 (目前使用第 {self.current_key_index + 1}/{len(self.finmind_keys_pool)} 把)")
        return True

    def _fetch_from_finmind(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        dbg.log(f"[{ticker}] 🌐 嘗試使用 [FinMind API] 抓取事件資料...")
        stock_id = ticker.replace(".TW", "").replace(".TWO", "")
        start_date = datetime.now().strftime("%Y-%m-%d")

        df_div = pd.DataFrame()
        df_earn = pd.DataFrame()

        # 決定最大重試次數：取網路重試次數與金鑰池大小的最大值
        max_attempts = max(self.MAX_RETRIES, len(self.finmind_keys_pool) if self.finmind_keys_pool else 1)
        attempt = 0

        # 加入 while 迴圈進行重試與輪替
        while attempt < max_attempts:
            params = {
                "dataset": "TaiwanStockDividend",
                "data_id": stock_id,
                "start_date": start_date
            }

            # 動態綁定目前生效的 Token
            if self.finmind_keys_pool:
                params["token"] = self.finmind_keys_pool[self.current_key_index]

            try:
                resp = requests.get(self.finmind_url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                # 攔截 Token 遭到限制的錯誤
                if data.get("msg") == "You have reached your daily limit" or data.get("status") == 403:
                    dbg.war(f"[{ticker}] ⚠️ FinMind Token 遭拒或已達上限 (回應碼: {data.get('status')})")

                    if self._rotate_key():
                        attempt += 1
                        continue # 帶著新金鑰重試
                    else:
                        break # 金鑰全數陣亡，準備交給外層觸發 yfinance 備援

                # 正常取得資料
                if data.get("msg") == StatusCol.SUCCESS:
                    data_list = data.get("data", [])

                    # 如果是空陣列 []，代表只是近期沒有除權息，這是正常的！
                    if not data_list:
                        dbg.log(f"[{ticker}] ℹ️ [FinMind API] 查詢成功，但目前無除權息事件。")
                        return pd.DataFrame(), pd.DataFrame()  # 正常回傳空表，絕不觸發斷路器

                    df_raw = pd.DataFrame(data_list)

                    if "CashExDividendTradingDate" not in df_raw.columns:
                        dbg.war(f"[{ticker}] FinMind 回傳資料缺失 CashExDividendTradingDate 欄位。")
                        return pd.DataFrame(), pd.DataFrame()

                    upcoming = df_raw[df_raw["CashExDividendTradingDate"] >= start_date].copy()

                    if not upcoming.empty:
                        if 'CashEarningsDistribution' in upcoming.columns:
                            cash_surplus = upcoming['CashEarningsDistribution'].fillna(0.0)
                        else:
                            dbg.log(f"[{ticker}] ℹ️ API 未提供 '盈餘分配' 欄位，系統自動視為 0.0")
                            cash_surplus = pd.Series(0.0, index=upcoming.index)

                        if 'CashCapitalReserve' in upcoming.columns:
                            cash_reserve = upcoming['CashCapitalReserve'].fillna(0.0)
                        else:
                            dbg.log(f"[{ticker}] ℹ️ API 未提供 '資本公積' 欄位，系統自動視為 0.0")
                            cash_reserve = pd.Series(0.0, index=upcoming.index)

                        total_cash_dividend = cash_surplus + cash_reserve

                        if (total_cash_dividend == 0.0).all():
                            dbg.war(f"[{ticker}] ⚠️ 警告：計算出的總現金股利為 0.0，請確認是否為純股票股利或 API 資料異常。")

                        df_div = pd.DataFrame({
                            'ticker': ticker,
                            'ex_date': upcoming["CashExDividendTradingDate"],
                            'cash_dividend': total_cash_dividend
                        })
                        dbg.log(f"[{ticker}] ✅ [FinMind API] 抓取除權息成功！")

                    return df_div, df_earn

                else:
                    dbg.war(f"[{ticker}] FinMind API 回傳格式異常: {data.get('msg')}")
                    return pd.DataFrame(), pd.DataFrame()

            except requests.exceptions.RequestException as e:
                dbg.war(f"[{ticker}] ❌ [FinMind API] 網路發生異常 (嘗試 {attempt + 1}/{max_attempts}): {e}")
                # 網路異常時，等待後重試
                if attempt < max_attempts - 1:
                    time.sleep(1) # 簡單的退避等待
                    attempt += 1
                    continue
                else:
                    break
            except Exception as e:
                dbg.error(f"[{ticker}] ❌ [FinMind API] 資料處理發生異常: {e}")
                break

        # 若跑出迴圈，代表 FinMind 徹底失敗
        return pd.DataFrame(), pd.DataFrame()

    def _fetch_from_yfinance(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        dbg.log(f"[{ticker}] 🛡️ 啟用備援機制，使用 [yfinance] 抓取事件資料...")
        df_div = pd.DataFrame()
        df_earn = pd.DataFrame()

        try:
            stock = yf.Ticker(ticker)
            today = datetime.now().strftime("%Y-%m-%d")

            # 抓除權息
            divs = stock.dividends
            if not divs.empty:
                df = divs.reset_index()
                df.columns = ['ex_date', 'cash_dividend']
                df['ex_date'] = df['ex_date'].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
                upcoming_div = df[df['ex_date'] >= today].copy()

                if not upcoming_div.empty:
                    upcoming_div['ticker'] = ticker
                    df_div = upcoming_div[['ticker', 'ex_date', 'cash_dividend']]

            dbg.log(f"[{ticker}] ✅ [yfinance] 備援抓取完成！")
            return df_div, df_earn
        except Exception as e:
            dbg.error(f"[{ticker}] ❌ [yfinance] 備援抓取亦失敗: {e}")
            return pd.DataFrame(), pd.DataFrame()