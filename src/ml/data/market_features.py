import re

import numpy as np
import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import MacroRawCol, MarketFeatureCol
from ml.params import IndicatorParams, MarketRiskCriteria


class MarketFeatureEngine:
    """
    大盤/總經大腦的特徵工程 (已升級：股、匯、債、期 四維雷達)。
    專注於合成台股大盤 (^TWII) 與美股費半 (^SOX) 的趨勢指標，並標記崩盤風險。
    """
    def __init__(self, lookahead: int, params: IndicatorParams = IndicatorParams(),
                 risk_criteria: MarketRiskCriteria = MarketRiskCriteria()):
        self.lookahead = lookahead
        self.params = params
        self.risk_criteria = risk_criteria

    def process_pipeline(self, df_market: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if df_market.empty: return pd.DataFrame()

        df_market.columns = [str(c).strip().lower() for c in df_market.columns]

        dbg.log("開始計算 LightGBM 大盤防禦特徵 (包含美債與期貨空單)...")
        data = df_market.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        # ==========================================
        # 1. 基礎台股大盤漲跌幅
        # ==========================================
        # twii_ret_1d = data[ai_vision_col].pct_change()

        # ==========================================
        # 2. 美股衍生特徵 (無未來函數對齊版)
        # ==========================================
        sox_close_col = self._get_ticker_col_name(MacroTicker.SOX)
        if sox_close_col in data.columns:
            # data[MarketFeatureCol.SOX_RET_1D] = data[sox_close_col].pct_change()
            data[MarketFeatureCol.SOX_RET_5D] = data[sox_close_col].pct_change(periods=5)
        else:
            # data[MarketFeatureCol.SOX_RET_1D] = 0.0
            data[MarketFeatureCol.SOX_RET_5D] = 0.0

        # ==========================================
        # 3. 台股大盤自身特徵 (Trend, Momentum & Volatility)
        # ==========================================
        ma_20 = data[ai_vision_col].rolling(window=self.params.MA_MONTH).mean()
        ma_60 = data[ai_vision_col].rolling(window=self.params.MA_QUARTER).mean()

        data[MarketFeatureCol.TWII_BIAS_20] = (data[ai_vision_col] - ma_20) / (ma_20 + 1e-9)
        data[MarketFeatureCol.TWII_BIAS_60] = (data[ai_vision_col] - ma_60) / (ma_60 + 1e-9)

        delta = data[ai_vision_col].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/self.params.RSI_PERIOD, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.params.RSI_PERIOD, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        data[MarketFeatureCol.TWII_RSI] = 100 - (100 / (1 + rs))

        ema_fast = data[ai_vision_col].ewm(span=self.params.MACD_FAST, adjust=False).mean()
        ema_slow = data[ai_vision_col].ewm(span=self.params.MACD_SLOW, adjust=False).mean()
        data[MarketFeatureCol.TWII_MACD] = (ema_fast - ema_slow) / (data[ai_vision_col] + 1e-9) * 100

        prev_close = data[ai_vision_col].shift(1)
        tr1 = data.get(StockCol.HIGH, data[ai_vision_col]) - data.get(StockCol.LOW, data[ai_vision_col])
        tr2 = (data.get(StockCol.HIGH, data[ai_vision_col]) - prev_close).abs()
        tr3 = (data.get(StockCol.LOW, data[ai_vision_col]) - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data[MarketFeatureCol.TWII_ATR_RATIO] = (true_range / (prev_close + 1e-9)).rolling(window=self.params.MA_WEEK).mean()

        # ==========================================
        # 4. 總經與籌碼特徵 (VIX, 匯率, 美債, 期貨空單)
        # ==========================================
        # [VIX 恐慌指數]
        vix_col = self._get_ticker_col_name(MacroTicker.VIX)
        if vix_col in data.columns:
            data[MarketFeatureCol.VIX_CLOSE] = data[vix_col]
            vix_ma20 = data[vix_col].rolling(20).mean()
            data[MarketFeatureCol.VIX_SURGE] = (data[vix_col] - vix_ma20) / (vix_ma20 + 1e-9)
        else:
            data[MarketFeatureCol.VIX_CLOSE] = 0.0
            data[MarketFeatureCol.VIX_SURGE] = 0.0

        # [台幣匯率]
        twd_col = self._get_ticker_col_name(MacroTicker.USDTWD)
        if twd_col in data.columns:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = data[twd_col].pct_change(periods=5)
        else:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = 0.0

        # 美國 10 年期公債殖利率 (US10Y)
        us10y_col = self._get_ticker_col_name(MacroTicker.US10Y)
        if us10y_col in data.columns:
            # 計算 5 日斜率 (殖利率短期狂飆是最危險的)
            data[MarketFeatureCol.US10Y_SURGE] = data[us10y_col].pct_change(periods=5)
        else:
            dbg.war("未偵測到美債殖利率 (^TNX) 資料，此特徵將補 0。")
            data[MarketFeatureCol.US10Y_SURGE] = 0.0

        # 外資台指期未平倉淨空單
        if MacroRawCol.FUTURES_NET_OI in data.columns:
            # 同時保留絕對水位，若低於 -20000 絕對是極度危險
            data[MarketFeatureCol.FUTURES_OI_LEVEL] = data[MacroRawCol.FUTURES_NET_OI]
        else:
            dbg.war("未偵測到外資期貨空單 (FUTURES_OI) 資料，此特徵將補 0。")
            data[MarketFeatureCol.FUTURES_OI_LEVEL] = 0.0

        # 散戶小台多空比 (RETAIL_LS_RATIO & RETAIL_LS_SURGE)
        if MacroRawCol.RETAIL_LS_RATIO in data.columns:
            data[MarketFeatureCol.RETAIL_LS_RATIO] = data[MacroRawCol.RETAIL_LS_RATIO]
            # 因為多空比有正有負，使用差分 (diff) 來算斜率比 pct_change 安全，不會因為除以負數而失真
            data[MarketFeatureCol.RETAIL_LS_SURGE] = data[MacroRawCol.RETAIL_LS_RATIO].diff(periods=3)
        else:
            dbg.war("未偵測到 'retail_ls_ratio' 欄位，散戶多空比特徵補 0")
            data[MarketFeatureCol.RETAIL_LS_RATIO] = 0.0
            data[MarketFeatureCol.RETAIL_LS_SURGE] = 0.0

        # 選擇權 Put/Call Ratio (PC_RATIO_CLOSE & PC_RATIO_BIAS_20)
        if MacroRawCol.PC_RATIO_CLOSE in data.columns:
            data[MarketFeatureCol.PC_RATIO_CLOSE] = data[MacroRawCol.PC_RATIO_CLOSE]
            # 計算 20 日均線乖離率
            pc_ma20 = data[MacroRawCol.PC_RATIO_CLOSE].rolling(window=20).mean()
            data[MarketFeatureCol.PC_RATIO_BIAS_20] = (data[MacroRawCol.PC_RATIO_CLOSE] - pc_ma20) / (pc_ma20 + 1e-9)
        else:
            dbg.war("未偵測到 'pc_ratio_close' 欄位，選擇權特徵補 0")
            data[MarketFeatureCol.PC_RATIO_CLOSE] = 0.0
            data[MarketFeatureCol.PC_RATIO_BIAS_20] = 0.0

        # 市場廣度背離
        if MacroRawCol.ADL_VALUE in data.columns:
            data[MacroRawCol.ADL_VALUE] = data[MacroRawCol.ADL_VALUE].ffill()
            twii_roc_10 = data[ai_vision_col].pct_change(periods=10)
            adl_roc_10 = data[MacroRawCol.ADL_VALUE].pct_change(periods=10)
            data[MarketFeatureCol.TWII_ADL_DIVERGENCE] = twii_roc_10 - adl_roc_10
        else:
            dbg.war("未偵測到 'adl_value' 欄位，廣度背離特徵補 0")
            data[MarketFeatureCol.TWII_ADL_DIVERGENCE] = 0.0

        # ==========================================
        # 5. 標籤：預測未來是否會有「大跌」 (Danger = 1)
        # ==========================================
        if is_training:
            # 取得未來的收盤價與季線 (60日)
            future_close = data[ai_vision_col].shift(-self.lookahead)
            future_ma60 = data[ai_vision_col].rolling(60).mean().shift(-self.lookahead)

            # 計算未來 N 天的累積漲跌幅
            future_ret_nd = data[ai_vision_col].pct_change(self.lookahead).shift(-self.lookahead)

            # 🟢 嚴苛化黑天鵝定義：
            # 條件 A (暴跌)：未來 N 天內，累積跌幅超過 4% (這在台指期代表超過 800 點的回檔)
            condition_a = (future_ret_nd < -0.04)

            # 條件 B (熊市)：未來收盤價跌破 60 日季線 (中期多空分水嶺)
            condition_b = (future_close < future_ma60)

            # 模型只會在「暴跌」或「跌破季線」時，才將其標記為危險 (Danger = 1)
            danger_condition = condition_a | condition_b

            data[MarketFeatureCol.TARGET_DANGER] = danger_condition.astype('Int64')
            data.loc[future_close.isna(), MarketFeatureCol.TARGET_DANGER] = pd.NA

        # ==========================================
        # 6. 清理回傳
        # ==========================================
        features = MarketFeatureCol.get_features()
        data = data.replace([np.inf, -np.inf], np.nan)

        if is_training:
            df_clean = data.dropna(subset=features + [MarketFeatureCol.TARGET_DANGER]).copy()
            df_clean[MarketFeatureCol.TARGET_DANGER] = df_clean[MarketFeatureCol.TARGET_DANGER].astype(int)
        else:
            df_clean = data.dropna(subset=features).copy()

        dbg.log(f"大盤特徵工程完成。產生 {len(df_clean)} 筆可用樣本。")
        return df_clean

    def _get_ticker_col_name(self, ticker: MacroTicker) -> str:
        """
        將 Ticker 統一轉換為乾淨的欄位名稱：
        """
        return f"{ticker.value.replace('^', '')}_close".lower()