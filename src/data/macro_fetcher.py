import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from const import GlobalCol
from data.const import StockCol
from data.params import DataLimit
from debug import dbg
from ml.const import MacroRawCol
from path import PathConfig

try:
    _ = yf.Ticker("SPY").history(period="1d")
except Exception:
    pass

class MacroFetcher:
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2

    def __init__(self) -> None:
        load_dotenv(PathConfig.ENV_FILE)
        self.finmind_token = os.getenv(GlobalCol.FINMIND_API_KEYS)

    def fetch_twse_adl_value(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        """
        [防禦升級] 利用 FinMind 的「櫃買報酬指數 (TPEx)」完美替代傳統 ADL 騰落指標。
        終極破案版：徹底拋棄不穩定的 yfinance，改用 FinMind 官方免費、零時區 Bug 的報酬指數表！
        來源: FinMind Open API (TaiwanStockTotalReturnIndex)
        """
        dbg.log("正在透過 FinMind API 獲取櫃買報酬指數 (TPEx) 建立大盤廣度背離指標...")

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data"

        params = {
            "dataset": "TaiwanStockTotalReturnIndex",  # 官方正宗大盤報酬指數資料集
            "data_id": "TPEx",                         # 精準指定：櫃買報酬指數
            "start_date": start_date,
            "token": self.finmind_token
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                # 取得資料陣列
                data_list = response.json().get("data", [])

                if not data_list:
                    continue

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

            except Exception as e:
                dbg.war(f"FinMind TPEx 指數抓取失敗 (嘗試 {attempt + 1}/{self.MAX_RETRIES}): {e}")

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_FACTOR ** attempt)

        dbg.error("❌ 櫃買報酬指數 (TPEx) 連線重試耗盡，放棄抓取。")
        return pd.DataFrame()

    def fetch_foreign_futures_oi(self, days: int = DataLimit.DAILY_DEFAULT_YEAR * 365) -> pd.DataFrame:
        """
        抓取外資「台指期 (TX) 未平倉淨部位 (Net Open Interest)」。
        負數代表外資滿手空單，是台股崩盤的最強領先指標。
        來源: FinMind Open API
        """
        dbg.log("正在透過 FinMind API 抓取外資台指期未平倉數據...")

        # 計算回推的起始日期
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanFuturesInstitutionalInvestors",
            "data_id": "TX", # TX 代表大台指期貨
            "start_date": start_date
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("msg") == "success" and len(data.get("data", [])) > 0:
                    df = pd.DataFrame(data["data"])
                    df_foreign = df[df['institutional_investors'].str.contains('外資|Foreign', case=False, na=False)].copy()

                    if df_foreign.empty:
                        investor_list = df['institutional_investors'].unique().tolist()
                        dbg.war(f"FinMind 回傳的投資人類別有: {investor_list}")
                        dbg.war("FinMind API 未回傳包含『外資』的期貨數據，請檢查上方印出的類別名稱。")
                        return pd.DataFrame()

                    # 計算「未平倉淨部位 (多單 - 空單)」
                    df_foreign[MacroRawCol.FUTURES_NET_OI] = (
                        df_foreign['long_open_interest_balance_volume'] -
                        df_foreign['short_open_interest_balance_volume']
                    )

                    # 格式化為與系統相容的 DataFrame
                    df_foreign['date'] = pd.to_datetime(df_foreign['date'])
                    df_foreign.set_index('date', inplace=True)
                    df_foreign.index.name = StockCol.DATE

                    # 只保留我們需要的核心欄位
                    df_clean = df_foreign[[MacroRawCol.FUTURES_NET_OI]].copy()

                    dbg.log(f"成功抓取外資期貨 OI 數據，共 {len(df_clean)} 筆。")
                    return df_clean

            except Exception as e:
                dbg.war(f"FinMind API 抓取失敗 (嘗試 {attempt + 1}/{self.MAX_RETRIES}): {e}")

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_FACTOR ** attempt)

        dbg.error("外資期貨 OI 抓取已達最大重試次數，放棄。")
        return pd.DataFrame()

    def fetch_retail_ls_ratio(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        """
        [防禦升級] 抓取「散戶小台指期多空比」。
        終極破案版：全市場改用 TaiwanFuturesDaily，完美支援 data_id="MTX"！
        來源: FinMind Open API
        """
        dbg.log("正在透過 FinMind API 抓取散戶小台期貨多空部位...")
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data"

        for attempt in range(self.MAX_RETRIES):
            try:
                # ==========================================
                # 1. 抓取「三大法人」小台指 (MTX) 未平倉部位
                # ==========================================
                res_inst = requests.get(url, params={
                    "dataset": "TaiwanFuturesInstitutionalInvestors",
                    "data_id": "MTX",
                    "start_date": start_date,
                    "token": self.finmind_token
                }, timeout=10)
                res_inst.raise_for_status()
                data_inst = res_inst.json().get("data", [])

                if not data_inst: continue

                df_inst = pd.DataFrame(data_inst)
                df_inst['date'] = pd.to_datetime(df_inst['date'])

                # 將三大法人 (外資、投信、自營商) 的多空未平倉量加總
                df_pivot = df_inst.pivot_table(
                    index='date',
                    values=['long_open_interest_balance_volume', 'short_open_interest_balance_volume'],
                    aggfunc='sum'
                ).fillna(0)

                legal_long = df_pivot['long_open_interest_balance_volume']
                legal_short = df_pivot['short_open_interest_balance_volume']

                # 法人淨多單 = 法人多單 - 法人空單
                # 因為期貨是零和遊戲，散戶淨多單 = -法人淨多單 = 法人空單 - 法人多單
                retail_net = legal_short - legal_long

                # ==========================================
                # 2. 抓取「全市場」小台指 (MTX) 總未平倉量 (Total Open Interest)
                # 修正：改用正宗 TaiwanFuturesDaily 資料集
                # ==========================================
                res_total = requests.get(url, params={
                    "dataset": "TaiwanFuturesDaily",  # 依據文檔精準對齊資料集名稱
                    "data_id": "MTX",                 # 帶入小台指代號
                    "start_date": start_date,
                    "token": self.finmind_token
                }, timeout=10)
                res_total.raise_for_status()
                data_total = res_total.json().get("data", [])

                if not data_total: continue

                df_total = pd.DataFrame(data_total)
                df_total['date'] = pd.to_datetime(df_total['date'])

                # TaiwanFuturesDaily 會回傳當天所有不同月份契約的明細
                # 直接 groupby('date') 並將 open_interest 加總，就是最精準的全市場總未平倉量！
                total_oi = df_total.groupby('date')['open_interest'].sum()

                # ==========================================
                # 3. 計算最終散戶多空比
                # ==========================================
                # 將兩個 Series 合併到同一個 DataFrame 中，透過 index(日期) 自動對齊
                df_res = pd.DataFrame({'retail_net': retail_net, 'total_oi': total_oi}).dropna()

                # 散戶小台多空比 = 散戶淨部位 / 全市場總未平倉量
                df_res[MacroRawCol.RETAIL_LS_RATIO] = df_res['retail_net'] / (df_res['total_oi'] + 1e-9)
                df_res.index.name = StockCol.DATE

                dbg.log(f"✅ 成功計算散戶小台多空比，共 {len(df_res)} 筆。")
                return df_res[[MacroRawCol.RETAIL_LS_RATIO]]

            except Exception as e:
                dbg.war(f"FinMind 散戶小台抓取失敗 (嘗試 {attempt + 1}/{self.MAX_RETRIES}): {e}")

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.BACKOFF_FACTOR ** attempt)

        return pd.DataFrame()

    def fetch_options_pc_ratio(self, days: int = DataLimit.MARKET_DEFAULT_YEAR * 365) -> pd.DataFrame:
        """
        [防禦升級] 抓取臺期所「臺指選擇權 Put/Call Ratio (未平倉量比率)」。
        終極防爆版：採用時間分段抓取法 (Chunked Fetching)，每次只抓 1 年，徹底解決巨量資料造成的 Read timed out。
        來源: FinMind Open API (TaiwanOptionDaily)
        """
        dbg.log(f"正在透過 FinMind API 分段抓取歷史選擇權日成交資料 (總計回推 {days} 天)...")

        # 設定時間分段的時間節點
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        # 以 365 天 (1 年) 為一期進行切分
        chunk_size = 365
        current_start = start_dt

        all_chunks = []
        url = "https://api.finmindtrade.com/api/v4/data"

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')

            dbg.log(f" ⏳ 正在分批撈取選擇權時段資料: {start_str} 至 {end_str} ...")

            # 參數對齊：與 fetch_retail_ls_ratio 一致，直接透過 params 傳遞 token
            params = {
                "dataset": "TaiwanOptionDaily", # 官方正宗選擇權日成交資料集
                "data_id": "TXO",               # 臺指選擇權
                "start_date": start_str,        # 分段起點
                "end_date": end_str,            # 分段終點
                "token": self.finmind_token     # 您的驗證密鑰
            }

            success_chunk = False
            for attempt in range(self.MAX_RETRIES):
                try:
                    # 將 params 傳入，並設定 20 秒寬裕逾時
                    response = requests.get(url, params=params, timeout=20)
                    response.raise_for_status()

                    data_list = response.json().get("data", [])
                    if data_list:
                        df_chunk = pd.DataFrame(data_list)
                        all_chunks.append(df_chunk)

                    success_chunk = True
                    break # 成功拿到這一批，跳出重試迴圈
                except Exception as e:
                    dbg.war(f" ⚠️ 選擇權時段 {start_str} 抓取嘗試失敗 ({attempt + 1}/{self.MAX_RETRIES}): {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.BACKOFF_FACTOR ** attempt)

            if not success_chunk:
                dbg.error(f"❌ 選擇權時段 {start_str} 達最大重試次數仍失敗，為確保數據連續性，中斷管線。")
                return pd.DataFrame()

            # 時間推進到下一段
            current_start = current_end + timedelta(days=1)

        if not all_chunks:
            dbg.war("⚠️ 全數時段跑完，但未取得任何選擇權歷史資料。")
            return pd.DataFrame()

        # ==========================================
        # 3. 本地端大團圓：合併所有時間碎片並計算指標
        # ==========================================
        try:
            df = pd.concat(all_chunks, ignore_index=True)
            df['date'] = pd.to_datetime(df['date'])

            # 頂級防護：直接用合約關鍵主鍵去重，100% 防止日夜盤重複加總且絕不誤殺
            df = df.drop_duplicates(subset=['date', 'option_id', 'strike_price', 'call_put'], keep='first')

            type_col = 'call_put' if 'call_put' in df.columns else 'type'
            if type_col not in df.columns:
                dbg.error(f"❌ 選擇權資料表缺失型態欄位，現有欄位: {df.columns.tolist()}")
                return pd.DataFrame()

            # 按日期與 買權(Call)/賣權(Put) 進行交叉轉置並將未平倉量 (open_interest) 加總
            df_pivot = df.pivot_table(
                index='date',
                columns=type_col,
                values='open_interest',
                aggfunc='sum'
            ).fillna(0)

            call_col = [c for c in df_pivot.columns if 'call' in str(c).lower() or '買' in str(c)]
            put_col = [c for c in df_pivot.columns if 'put' in str(c).lower() or '賣' in str(c)]

            if not call_col or not put_col:
                dbg.war(f"無法識別選擇權多空欄位，現有欄位: {df_pivot.columns.tolist()}")
                return pd.DataFrame()

            total_call = df_pivot[call_col[0]]
            total_put = df_pivot[put_col[0]]

            # 公式：Put/Call Ratio = 賣權未平倉量 / 買權未平倉量
            df_res = pd.DataFrame(index=df_pivot.index)
            df_res[MacroRawCol.PC_RATIO_CLOSE] = total_put / (total_call + 1e-9)
            df_res.index.name = StockCol.DATE

            dbg.log(f"✅ 選擇權歷史分段資料成功大團圓！共合成 {len(df_res)} 筆 P/C Ratio 數據。")
            return df_res[[MacroRawCol.PC_RATIO_CLOSE]]

        except Exception as e:
            dbg.error(f"本地端串接/合成選擇權指標時發生異常: {e}")
            return pd.DataFrame()
