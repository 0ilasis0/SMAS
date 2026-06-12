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
        if not self._finmind_circuit_breaker_tripped:
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

        # 4. 決定最大重試次數：取網路重試次數與金鑰池大小的最大值
        max_attempts = max(self.MAX_RETRIES, len(self.finmind_keys_pool) if self.finmind_keys_pool else 1)
        attempt = 0

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
                if data.get("msg") == StatusCol.SUCCESS and data.get("data"):
                    df_raw = pd.DataFrame(data["data"])

                    if "CashExDividendTradingDate" not in df_raw.columns:
                        dbg.war(f"[{ticker}] FinMind 回傳資料缺失 CashExDividendTradingDate 欄位。")
                        return pd.DataFrame(), pd.DataFrame()

                    upcoming = df_raw[df_raw["CashExDividendTradingDate"] >= start_date].copy()

                    if not upcoming.empty:
                        cash_surplus = upcoming.get('CashEarningsDistribution', 0.0).fillna(0.0)
                        cash_reserve = upcoming.get('CashCapitalReserve', 0.0).fillna(0.0)
                        total_cash_dividend = cash_surplus + cash_reserve

                        df_div = pd.DataFrame({
                            'ticker': ticker,
                            'ex_date': upcoming["CashExDividendTradingDate"],
                            'cash_dividend': total_cash_dividend
                        })
                        dbg.log(f"[{ticker}] ✅ [FinMind API] 抓取除權息成功！")

                    return df_div, df_earn

                else:
                    # 其他非 Token 的 API 格式異常
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

        return pd.DataFrame(), pd.DataFrame()

    def _fetch_from_finmind(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        dbg.log(f"[{ticker}] 🌐 嘗試使用 [FinMind API] 抓取事件資料...")
        stock_id = ticker.replace(".TW", "").replace(".TWO", "")
        start_date = datetime.now().strftime("%Y-%m-%d")

        df_div = pd.DataFrame()
        df_earn = pd.DataFrame()

        try:
            params = {
                "dataset": "TaiwanStockDividend",
                "data_id": stock_id,
                "start_date": start_date,
                "token": self.finmind_token
            }
            resp = requests.get(self.finmind_url, params=params, timeout=10)
            data = resp.json()

            if data.get("msg") == StatusCol.SUCCESS and data.get("data"):
                df_raw = pd.DataFrame(data["data"])

                # 確保除息日欄位存在，否則放棄
                if "CashExDividendTradingDate" not in df_raw.columns:
                    dbg.war(f"[{ticker}] FinMind 回傳資料缺失 CashExDividendTradingDate 欄位。")
                    return pd.DataFrame(), pd.DataFrame()

                # 篩選未來的除息日
                upcoming = df_raw[df_raw["CashExDividendTradingDate"] >= start_date].copy()

                if not upcoming.empty:
                    # 總現金股利 = 盈餘分配 + 資本公積，如果該欄位不存在或為空 (NaN)，則補 0.0 後相加
                    cash_surplus = upcoming.get('CashEarningsDistribution', 0.0).fillna(0.0)
                    cash_reserve = upcoming.get('CashCapitalReserve', 0.0).fillna(0.0)
                    total_cash_dividend = cash_surplus + cash_reserve

                    df_div = pd.DataFrame({
                        'ticker': ticker,
                        'ex_date': upcoming["CashExDividendTradingDate"],
                        'cash_dividend': total_cash_dividend
                    })
                    dbg.log(f"[{ticker}] ✅ [FinMind API] 抓取除權息成功！")

            return df_div, df_earn

        except Exception as e:
            dbg.error(f"[{ticker}] ❌ [FinMind API] 發生異常: {e}")
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