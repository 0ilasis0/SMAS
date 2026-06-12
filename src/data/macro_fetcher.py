import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from base import KeyManager
from data.const import StockCol
from data.params import DataLimit
from debug import dbg
from ml.const import MacroRawCol

try:
    _ = yf.Ticker("SPY").history(period="1d")
except Exception:
    pass

class MacroFetcher:
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self) -> None:
        self.current_key_index = 0

        self.finmind_keys_pool = KeyManager.get_finmind_keys()
        if not self.finmind_keys_pool:
            dbg.war("⚠️ 找不到任何 FinMind 金鑰，將以無金鑰模式運行 (受限速保護)。")

    def _rotate_key(self) -> bool:
        """
        切換到下一把金鑰。
        回傳 True 代表切換成功；回傳 False 代表所有金鑰都用盡了。
        """
        if not self.finmind_keys_pool:
            return False

        self.current_key_index += 1

        # 如果索引超出清單長度，代表整把鑰匙串都試過了
        if self.current_key_index >= len(self.finmind_keys_pool):
            dbg.error("❌ 所有 FinMind 金鑰皆已耗盡或遭到限制！本次管線將中斷。")
            return False

        dbg.log(f"🔄 偵測到 FinMind Token 受限，已自動切換至備用金鑰 (目前使用第 {self.current_key_index + 1}/{len(self.finmind_keys_pool)} 把)")
        return True

    # 共用的底層 API 抓取與重試引擎
    def _fetch_from_finmind(self, params: dict, api_name: str, timeout: int = 10) -> list:
        """
        統一處理 FinMind API 的連線、重試、Token 輪替與資料提取邏輯。
        回傳 JSON 內的 'data' 陣列；若徹底失敗則回傳空陣列 []。
        """
        # 決定最大重試次數：取預設網路重試次數與金鑰池大小的最大值
        max_attempts = max(self.MAX_RETRIES, len(self.finmind_keys_pool) if self.finmind_keys_pool else 1)
        attempt = 0

        while attempt < max_attempts:
            # 動態綁定目前生效的 Token (如果在無金鑰模式，則留空)
            if self.finmind_keys_pool:
                params["token"] = self.finmind_keys_pool[self.current_key_index]

            try:
                response = requests.get(self.FINMIND_URL, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()

                if data.get("msg") == "You have reached your daily limit" or data.get("status") == 403:
                    dbg.war(f"[{api_name}] ⚠️ Token 遭拒或已達上限 (回應碼: {data.get('status')})")

                    if self._rotate_key():
                        attempt += 1
                        continue # 帶著新金鑰，重新發送請求
                    else:
                        break # 金鑰全數陣亡，跳出迴圈

                # 正常取得資料
                elif data.get("msg") == "success" or "data" in data:
                    return data.get("data", [])

                # 其他 API 格式異常
                else:
                    dbg.war(f"[{api_name}] API 回傳格式異常: {data}")
                    return []

            except requests.exceptions.RequestException as e:
                dbg.war(f"[{api_name}] 網路連線或讀取失敗 (嘗試 {attempt + 1}/{max_attempts}): {e}")
                # 網路異常時，觸發時間退避 (Exponential Backoff) 而不急著換金鑰
                if attempt < max_attempts - 1:
                    time.sleep(self.BACKOFF_FACTOR ** attempt)
                    attempt += 1
                    continue
                else:
                    break

        dbg.error(f"❌ [{api_name}] 連線重試與金鑰皆耗盡，放棄抓取。")
        return []

    # ==========================================
    # 業務函數 (大幅瘦身，專注於資料處理)
    # ==========================================
    def fetch_twse_adl_value(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        dbg.log("正在透過 FinMind API 獲取櫃買報酬指數 (TPEx) 建立大盤廣度背離指標...")
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        params = {
            "dataset": "TaiwanStockTotalReturnIndex",
            "data_id": "TPEx",
            "start_date": start_date
        }

        # 呼叫共用引擎
        data_list = self._fetch_from_finmind(params, "櫃買報酬指數 (TPEx)")
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index.name = StockCol.DATE

        price_col = 'price' if 'price' in df.columns else 'close'
        if price_col not in df.columns:
            dbg.error(f"❌ TPEx 資料集缺失價格欄位，現有欄位: {df.columns.tolist()}")
            return pd.DataFrame()

        df_res = pd.DataFrame(index=df.index)
        df_res[MacroRawCol.ADL_VALUE] = df[price_col].astype(float)

        dbg.log(f"✅ 成功載入櫃買報酬指數 (TPEx)，共 {len(df_res)} 筆交易日。")
        return df_res[[MacroRawCol.ADL_VALUE]]

    def fetch_foreign_futures_oi(self, days: int = DataLimit.DAILY_DEFAULT_YEAR * 365) -> pd.DataFrame:
        dbg.log("正在透過 FinMind API 抓取外資台指期未平倉數據...")
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        params = {
            "dataset": "TaiwanFuturesInstitutionalInvestors",
            "data_id": "TX",
            "start_date": start_date
        }

        # 呼叫共用引擎
        data_list = self._fetch_from_finmind(params, "外資期貨 OI")
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        df_foreign = df[df['institutional_investors'].str.contains('外資|Foreign', case=False, na=False)].copy()

        if df_foreign.empty:
            investor_list = df['institutional_investors'].unique().tolist()
            dbg.war(f"FinMind API 未回傳包含『外資』的期貨數據。現有類別: {investor_list}")
            return pd.DataFrame()

        df_foreign[MacroRawCol.FUTURES_NET_OI] = (
            df_foreign['long_open_interest_balance_volume'] - df_foreign['short_open_interest_balance_volume']
        )

        df_foreign['date'] = pd.to_datetime(df_foreign['date'])
        df_foreign.set_index('date', inplace=True)
        df_foreign.index.name = StockCol.DATE

        df_clean = df_foreign[[MacroRawCol.FUTURES_NET_OI]].copy()
        dbg.log(f"✅ 成功抓取外資期貨 OI 數據，共 {len(df_clean)} 筆。")
        return df_clean

    def fetch_retail_ls_ratio(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        dbg.log("正在透過 FinMind API 抓取散戶小台期貨多空部位...")
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # 1. 抓取「三大法人」小台指 (MTX) 未平倉部位
        data_inst = self._fetch_from_finmind({
            "dataset": "TaiwanFuturesInstitutionalInvestors",
            "data_id": "MTX",
            "start_date": start_date
        }, "三大法人小台指")
        if not data_inst: return pd.DataFrame()

        df_inst = pd.DataFrame(data_inst)
        df_inst['date'] = pd.to_datetime(df_inst['date'])
        df_pivot = df_inst.pivot_table(
            index='date',
            values=['long_open_interest_balance_volume', 'short_open_interest_balance_volume'],
            aggfunc='sum'
        ).fillna(0)

        legal_long = df_pivot['long_open_interest_balance_volume']
        legal_short = df_pivot['short_open_interest_balance_volume']
        retail_net = legal_short - legal_long

        # 2. 抓取「全市場」小台指 (MTX) 總未平倉量
        data_total = self._fetch_from_finmind({
            "dataset": "TaiwanFuturesDaily",
            "data_id": "MTX",
            "start_date": start_date
        }, "全市場小台指")
        if not data_total: return pd.DataFrame()

        df_total = pd.DataFrame(data_total)
        df_total['date'] = pd.to_datetime(df_total['date'])
        total_oi = df_total.groupby('date')['open_interest'].sum()

        # 3. 計算最終散戶多空比
        df_res = pd.DataFrame({'retail_net': retail_net, 'total_oi': total_oi}).dropna()
        df_res[MacroRawCol.RETAIL_LS_RATIO] = df_res['retail_net'] / (df_res['total_oi'] + 1e-9)
        df_res.index.name = StockCol.DATE

        dbg.log(f"✅ 成功計算散戶小台多空比，共 {len(df_res)} 筆。")
        return df_res[[MacroRawCol.RETAIL_LS_RATIO]]

    def fetch_options_pc_ratio(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        dbg.log(f"正在透過 FinMind API 分段抓取歷史選擇權日成交資料 (總計回推 {days} 天)...")

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        chunk_size = 365
        current_start = start_dt
        all_chunks = []

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')

            dbg.log(f" ⏳ 正在分批撈取選擇權時段資料: {start_str} 至 {end_str} ...")

            params = {
                "dataset": "TaiwanOptionDaily",
                "data_id": "TXO",
                "start_date": start_str,
                "end_date": end_str
            }

            # 呼叫共用引擎 (設定 20 秒較長超時)
            data_list = self._fetch_from_finmind(params, f"選擇權 ({start_str})", timeout=20)

            if not data_list:
                dbg.error(f"❌ 選擇權時段 {start_str} 抓取失敗，中斷管線。")
                return pd.DataFrame()

            all_chunks.append(pd.DataFrame(data_list))
            current_start = current_end + timedelta(days=1)

        if not all_chunks:
            return pd.DataFrame()

        # 本地端計算
        try:
            df = pd.concat(all_chunks, ignore_index=True)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop_duplicates(subset=['date', 'option_id', 'strike_price', 'call_put'], keep='first')

            type_col = 'call_put' if 'call_put' in df.columns else 'type'
            if type_col not in df.columns:
                dbg.error(f"❌ 選擇權資料表缺失型態欄位: {df.columns.tolist()}")
                return pd.DataFrame()

            df_pivot = df.pivot_table(index='date', columns=type_col, values='open_interest', aggfunc='sum').fillna(0)

            call_col = [c for c in df_pivot.columns if 'call' in str(c).lower() or '買' in str(c)]
            put_col = [c for c in df_pivot.columns if 'put' in str(c).lower() or '賣' in str(c)]

            if not call_col or not put_col:
                dbg.war(f"無法識別選擇權多空欄位: {df_pivot.columns.tolist()}")
                return pd.DataFrame()

            total_call = df_pivot[call_col[0]]
            total_put = df_pivot[put_col[0]]

            df_res = pd.DataFrame(index=df_pivot.index)
            df_res[MacroRawCol.PC_RATIO_CLOSE] = total_put / (total_call + 1e-9)
            df_res.index.name = StockCol.DATE

            dbg.log(f"✅ 選擇權歷史分段資料成功大團圓！共合成 {len(df_res)} 筆 P/C Ratio 數據。")
            return df_res[[MacroRawCol.PC_RATIO_CLOSE]]

        except Exception as e:
            dbg.error(f"本地端串接/合成選擇權指標時發生異常: {e}")
            return pd.DataFrame()
