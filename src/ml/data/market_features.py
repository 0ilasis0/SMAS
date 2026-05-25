import numpy as np
import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import MarketFeatureCol
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
        twii_ret_1d = data[ai_vision_col].pct_change()

        # ==========================================
        # 2. 美股衍生特徵 (無未來函數對齊版)
        # ==========================================
        sox_close_col = f"{MacroTicker.SOX.value.replace('^', '')}_close".lower()
        if sox_close_col in data.columns:
            data[MarketFeatureCol.SOX_RET_1D] = data[sox_close_col].pct_change()
            data[MarketFeatureCol.SOX_RET_5D] = data[sox_close_col].pct_change(periods=5)
            data[MarketFeatureCol.SOX_TWII_SPREAD] = twii_ret_1d - data[MarketFeatureCol.SOX_RET_1D]
        else:
            data[MarketFeatureCol.SOX_RET_1D] = 0.0
            data[MarketFeatureCol.SOX_RET_5D] = 0.0
            data[MarketFeatureCol.SOX_TWII_SPREAD] = 0.0

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

        # 成交量取 Log 差分，消除極端節日效應
        vol_col = str(StockCol.VOLUME)
        data[MarketFeatureCol.TWII_VOL_CHG] = np.log1p(data[vol_col]) - np.log1p(data[vol_col].shift(1))

        prev_close = data[ai_vision_col].shift(1)
        tr1 = data.get(StockCol.HIGH, data[ai_vision_col]) - data.get(StockCol.LOW, data[ai_vision_col])
        tr2 = (data.get(StockCol.HIGH, data[ai_vision_col]) - prev_close).abs()
        tr3 = (data.get(StockCol.LOW, data[ai_vision_col]) - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data[MarketFeatureCol.TWII_ATR_RATIO] = (true_range / (prev_close + 1e-9)).rolling(window=self.params.MA_WEEK).mean()

        # === 大盤 K 線型態 (判斷大盤恐慌下殺或強勢軋空) ===
        max_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].max(axis=1)
        min_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].min(axis=1)
        price_range = (data.get(StockCol.HIGH, data[StockCol.CLOSE]) - data.get(StockCol.LOW, data[StockCol.CLOSE])).clip(lower=0.01)

        data[MarketFeatureCol.TWII_K_UPPER] = (data[StockCol.HIGH] - max_open_close) / price_range
        data[MarketFeatureCol.TWII_K_LOWER] = (min_open_close - data[StockCol.LOW]) / price_range
        data[MarketFeatureCol.TWII_K_BODY] = (data[StockCol.CLOSE] - data.get(StockCol.OPEN, data[StockCol.CLOSE])) / price_range

        # ==========================================
        # 4. 總經與籌碼特徵 (VIX, 匯率, 美債, 期貨空單)
        # ==========================================
        # [VIX 恐慌指數]
        vix_col = f"{MacroTicker.VIX.value.replace('^', '')}_close".lower()
        if vix_col in data.columns:
            data[MarketFeatureCol.VIX_CLOSE] = data[vix_col]
            vix_ma20 = data[vix_col].rolling(20).mean()
            data[MarketFeatureCol.VIX_SURGE] = (data[vix_col] - vix_ma20) / (vix_ma20 + 1e-9)
        else:
            data[MarketFeatureCol.VIX_CLOSE] = 0.0
            data[MarketFeatureCol.VIX_SURGE] = 0.0

        # [台幣匯率]
        twd_col = f"{MacroTicker.USDTWD.value}_close".lower()
        if twd_col in data.columns:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = data[twd_col].pct_change(periods=5)
        else:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = 0.0

        # 美國 10 年期公債殖利率 (US10Y)
        # 捕捉全球資金緊縮、科技股估值下殺的系統性風險
        us10y_col = "^tnx_close" # Yahoo Finance 的代號通常是 ^TNX
        if us10y_col in data.columns:
            # 計算 5 日斜率 (殖利率短期狂飆是最危險的)
            data[MarketFeatureCol.US10Y_SURGE] = data[us10y_col].pct_change(periods=5)
        else:
            dbg.war("未偵測到美債殖利率 (^TNX) 資料，此特徵將補 0。")
            data[MarketFeatureCol.US10Y_SURGE] = 0.0

        # 外資台指期未平倉淨空單 (Futures OI)
        # 捕捉外資假拉抬、真佈空的台股專屬核彈級前兆
        futures_oi_col = "futures_oi" # 假設 DataManager 讀取出來的欄位名為此
        if futures_oi_col in data.columns:
            # 同時保留絕對水位，若低於 -20000 絕對是極度危險
            data[MarketFeatureCol.FUTURES_OI_LEVEL] = data[futures_oi_col]
        else:
            dbg.war("未偵測到外資期貨空單 (FUTURES_OI) 資料，此特徵將補 0。")
            data[MarketFeatureCol.FUTURES_OI_LEVEL] = 0.0

        # ==========================================
        # 5. 標籤：預測未來是否會有「大跌」 (Danger = 1)
        # ==========================================
        if is_training:
            future_close = data[ai_vision_col].shift(-self.lookahead)
            future_ma20 = data[ai_vision_col].rolling(20).mean().shift(-self.lookahead)
            future_ret_5d = data[ai_vision_col].pct_change(5).shift(-self.lookahead)

            danger_condition = (future_close < future_ma20) | (future_ret_5d < -0.01)

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