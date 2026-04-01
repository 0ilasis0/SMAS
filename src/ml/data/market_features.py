import numpy as np
import pandas as pd

from data.const import StockCol
from debug import dbg
from ml.const import MarketFeatureCol


class MarketFeatureEngine:
    """
    大盤/總經大腦的特徵工程。
    專注於合成台股大盤 (^TWII) 與美股費半 (^SOX) 的趨勢指標，並標記崩盤風險。
    """
    def __init__(self, lookahead: int):
        self.lookahead = lookahead

    def process_pipeline(self, df_market: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if df_market.empty: return pd.DataFrame()

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
        if MarketFeatureCol.SOX_CLOSE in data.columns:
            data[MarketFeatureCol.SOX_RET_1D] = data[MarketFeatureCol.SOX_CLOSE].pct_change()
            data[MarketFeatureCol.SOX_RET_5D] = data[MarketFeatureCol.SOX_CLOSE].pct_change(5)
            # 台美相對強弱差 (Spread)。若費半暴跌但台股抗跌，數值會大於 0
            data[MarketFeatureCol.SOX_TWII_SPREAD] = twii_ret_1d - data[MarketFeatureCol.SOX_RET_1D]
        else:
            dbg.war("警告：未發現 sox_close 欄位，費半相關特徵將補 0。")
            data[MarketFeatureCol.SOX_RET_1D] = 0.0
            data[MarketFeatureCol.SOX_RET_5D] = 0.0
            data[MarketFeatureCol.SOX_TWII_SPREAD] = 0.0

        # ==========================================
        # 3. 台股大盤自身特徵 (Trend, Momentum & Volatility)
        # ==========================================
        ma_20 = data[ai_vision_col].rolling(window=20).mean()
        ma_60 = data[ai_vision_col].rolling(window=60).mean()

        data[MarketFeatureCol.TWII_BIAS_20] = (data[ai_vision_col] - ma_20) / ma_20
        data[MarketFeatureCol.TWII_BIAS_60] = (data[ai_vision_col] - ma_60) / ma_60

        # RSI
        delta = data[ai_vision_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        data[MarketFeatureCol.TWII_RSI] = 100 - (100 / (1 + rs))

        # MACD (PPO)
        ema_12 = data[ai_vision_col].ewm(span=12, adjust=False).mean()
        ema_26 = data[ai_vision_col].ewm(span=26, adjust=False).mean()
        data[MarketFeatureCol.TWII_MACD] = (ema_12 - ema_26) / data[ai_vision_col] * 100

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

        # 取 5 日平均，並除以昨收轉為百分比
        data[MarketFeatureCol.TWII_ATR_RATIO] = (true_range / prev_close).rolling(window=5).mean()

        # ==========================================
        # 4. 標籤：預測未來是否會有「大跌」 (Danger = 1)
        # ==========================================
        if is_training:
            adj_factor = data[ai_vision_col] / (data[StockCol.CLOSE] + 1e-9)
            adj_low = data[StockCol.LOW] * adj_factor
            # 未來 5 天內的最低點
            future_low_min = adj_low.rolling(window=self.lookahead, min_periods=1).min().shift(-self.lookahead)
            # 如果未來 5 天內會跌破現在收盤價的 2.5%，標記為危險(大盤跌 2.5% 等同於個股跌 5%~10%，這是一個非常嚴重的崩盤警訊)
            danger_condition = future_low_min < (data[ai_vision_col] * 0.975)

            data[MarketFeatureCol.TARGET_DANGER] = danger_condition.astype('Int64')
            data.loc[future_low_min.isna(), MarketFeatureCol.TARGET_DANGER] = pd.NA

        # ==========================================
        # 5. 清理回傳
        # ==========================================
        features = MarketFeatureCol.get_features()
        data = data.replace([np.inf, -np.inf], np.nan)

        if is_training:
            df_clean = data.dropna(subset=features + [MarketFeatureCol.TARGET_DANGER])
            df_clean[MarketFeatureCol.TARGET_DANGER] = df_clean[MarketFeatureCol.TARGET_DANGER].astype(int)
        else:
            df_clean = data.dropna(subset=features)

        dbg.log(f"大盤特徵工程完成。產生 {len(df_clean)} 筆可用樣本。")
        return df_clean
