import numpy as np
import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import MarketFeatureCol
from ml.params import IndicatorParams, MarketRiskCriteria


class MarketFeatureEngine:
    """
    大盤/總經大腦的特徵工程。
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

        dbg.log("開始計算 LightGBM 大盤防禦特徵...")
        data = df_market.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        # ==========================================
        # 1. 基礎台股大盤漲跌幅 (供後續相對強弱計算)
        # ==========================================
        twii_ret_1d = data[ai_vision_col].pct_change()

        # ==========================================
        # 2. 美股衍生特徵 (無未來函數對齊版)
        # ==========================================
        sox_close_col = f"{MacroTicker.SOX.value.replace('^', '')}_close".lower()
        if sox_close_col in data.columns:
            data[MarketFeatureCol.SOX_RET_1D] = data[sox_close_col].pct_change()
            data[MarketFeatureCol.SOX_RET_5D] = data[sox_close_col].pct_change(periods=5)
            # 台美相對強弱差 (Spread)。若費半暴跌但台股抗跌，數值會大於 0
            data[MarketFeatureCol.SOX_TWII_SPREAD] = twii_ret_1d - data[MarketFeatureCol.SOX_RET_1D]
        else:
            dbg.war(f"警告：未發現 {sox_close_col} 欄位，費半相關特徵將補 0。")
            dbg.log(f"目前可用的欄位有: {data.columns.tolist()}")
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

        # 使用 True Range (真實區間) 取代單純的高低價差，完美捕捉跳空恐慌
        prev_close = data[ai_vision_col].shift(1)
        tr1 = data[StockCol.HIGH] - data[StockCol.LOW]
        tr2 = (data[StockCol.HIGH] - prev_close).abs()
        tr3 = (data[StockCol.LOW] - prev_close).abs()

        # 沿著 columns 方向取最大值，算出每一天的 True Range
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        data[MarketFeatureCol.TWII_ATR_RATIO] = (true_range / (prev_close + 1e-9)).rolling(window=self.params.MA_WEEK).mean()

        # === 大盤 K 線型態 (判斷大盤恐慌下殺或強勢軋空) ===
        max_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].max(axis=1)
        min_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].min(axis=1)
        price_range = (data[StockCol.HIGH] - data[StockCol.LOW]).clip(lower=0.01)

        data[MarketFeatureCol.TWII_K_UPPER] = (data[StockCol.HIGH] - max_open_close) / price_range
        data[MarketFeatureCol.TWII_K_LOWER] = (min_open_close - data[StockCol.LOW]) / price_range
        data[MarketFeatureCol.TWII_K_BODY] = (data[StockCol.CLOSE] - data[StockCol.OPEN]) / price_range

        # === VIX 恐慌指數 ===
        # 假設您的 DB 會把 ^VIX 變成 VIX_close
        vix_col = f"{MacroTicker.VIX.value.replace('^', '')}_close".lower()
        if vix_col in data.columns:
            data[MarketFeatureCol.VIX_CLOSE] = data[vix_col]
            vix_ma20 = data[vix_col].rolling(20).mean()
            data[MarketFeatureCol.VIX_SURGE] = (data[vix_col] - vix_ma20) / (vix_ma20 + 1e-9)
        else:
            data[MarketFeatureCol.VIX_CLOSE] = 0.0
            data[MarketFeatureCol.VIX_SURGE] = 0.0

        # === 台幣匯率 (外資動向) ===
        # 假設您的 DB 會把 TWD=X 變成 TWD=X_close
        twd_col = f"{MacroTicker.USDTWD.value}_close".lower()
        if twd_col in data.columns:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = data[twd_col].pct_change(periods=5)
        else:
            data[MarketFeatureCol.TWD_DEPRECIATION_5D] = 0.0

        # ==========================================
        # 4. 標籤：預測未來是否會有「大跌」 (Danger = 1)
        # ==========================================
        if is_training:
            # 取得還原權值 (防止大盤除權息導致的假跌幅)
            adj_factor = data[ai_vision_col] / (data[StockCol.CLOSE] + 1e-9)
            adj_low = data[StockCol.LOW] * adj_factor

            # 未來 N 天內的最低點 (MAE)
            future_low_min = adj_low.rolling(window=self.lookahead, min_periods=1).min().shift(-self.lookahead)

            # 計算大盤專屬的動態停損線
            # 我們可以直接利用前面已經算好的 true_range
            atr = true_range.rolling(window=self.risk_criteria.ATR_LOOKBACK).mean()

            # 門檻：跌破 [目前收盤價 - (1.5倍大盤ATR)]
            danger_price_threshold = data[ai_vision_col] - (atr * self.risk_criteria.CRASH_THRESHOLD_ATR)

            # 判定危險：只要未來最低點跌破這個動態門檻，標記為 Danger (1)
            danger_condition = future_low_min < danger_price_threshold

            data[MarketFeatureCol.TARGET_DANGER] = danger_condition.astype('Int64')
            data.loc[future_low_min.isna(), MarketFeatureCol.TARGET_DANGER] = pd.NA

        # ==========================================
        # 5. 清理回傳
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
