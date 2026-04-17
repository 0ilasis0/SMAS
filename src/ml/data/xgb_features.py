import numpy as np
import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.params import IndicatorParams


class XGBFeatureEngine:
    """
    以 XGBoost 設計的特徵工程。
    負責計算技術指標 (MA, RSI, MACD) 與生成預測標籤 (Target)。
    """
    def __init__(self, params: IndicatorParams = IndicatorParams()):
        self.params = params

    def process_pipeline(self, df: pd.DataFrame, lookahead: int, is_training: bool = True) -> pd.DataFrame:
        """執行完整的 XGBoost 特徵管線"""
        if df.empty:
            dbg.war("輸入的 DataFrame 為空，跳過特徵工程。")
            return df

        dbg.log("開始計算 XGBoost 技術特徵與標籤...")

        df_features = self._create_daily_features(df)
        df_labeled = self._create_labels(df_features, lookahead)

        features = FeatureCol.get_features()

        df_labeled = df_labeled.replace([np.inf, -np.inf], np.nan)

        if is_training:
            df_clean = df_labeled.dropna(subset=features + [FeatureCol.TARGET])
        else:
            df_clean = df_labeled.dropna(subset=features)

        initial_len = len(df_labeled)
        final_len = len(df_clean)
        dbg.log(f"特徵工程完成。移除了 {initial_len - final_len} 筆含 NaN 的無效資料，剩餘 {final_len} 筆可用樣本。")
        return df_clean

    def _create_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算波段 (日 K) 特徵"""
        if df.empty: return df

        data = df.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        ma_w = data[ai_vision_col].rolling(window=self.params.MA_WEEK).mean()
        ma_m = data[ai_vision_col].rolling(window=self.params.MA_MONTH).mean()
        ma_q = data[ai_vision_col].rolling(window=self.params.MA_QUARTER).mean()
        ma_y = data[ai_vision_col].rolling(window=self.params.MA_YEAR).mean()

        data[FeatureCol.BIAS_WEEK] = (data[ai_vision_col] - ma_w) / (ma_w + 1e-9)
        data[FeatureCol.BIAS_MONTH] = (data[ai_vision_col] - ma_m) / (ma_m + 1e-9)
        data[FeatureCol.BIAS_QUARTER] = (data[ai_vision_col] - ma_q) / (ma_q + 1e-9)
        data[FeatureCol.BIAS_YEAR] = (data[ai_vision_col] - ma_y) / (ma_y + 1e-9)

        # 使用更準確的 EMA 來計算 RSI，增加指標對近期價格的敏感度
        delta = data[ai_vision_col].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/self.params.RSI_PERIOD, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.params.RSI_PERIOD, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        rsi_100 = 100 - (100 / (1 + rs))
        data[FeatureCol.RSI] = (rsi_100 - 50) / 50.0

        ema_fast = data[ai_vision_col].ewm(span=self.params.MACD_FAST, adjust=False).mean()
        ema_slow = data[ai_vision_col].ewm(span=self.params.MACD_SLOW, adjust=False).mean()
        data[FeatureCol.MACD] = (ema_fast - ema_slow) / (data[ai_vision_col] + 1e-9) * 100
        data[FeatureCol.MACD_SIGNAL] = data[FeatureCol.MACD].ewm(span=self.params.MACD_SIGNAL, adjust=False).mean()

        # 布林通道寬度 (BB Width) - 抓波動率壓縮
        rolling_std = data[ai_vision_col].rolling(window=self.params.MA_MONTH).std()
        data[FeatureCol.BB_WIDTH] = (rolling_std * 2) / (ma_m + 1e-9)

        price_diff = data[ai_vision_col].diff()
        direction = np.sign(price_diff)
        # 處理平盤 (direction=0) 的情況，保持成交量不增不減
        direction = direction.fillna(0)
        # OBV 是絕對數值非常大，我們把它轉換為 20 日移動平均乖離率，讓 AI 更好吸收
        raw_obv = (direction * data[StockCol.VOLUME]).cumsum()
        obv_ma20 = raw_obv.rolling(window=20).mean()
        # 加上 1 防止分母為 0
        data[FeatureCol.OBV] = (raw_obv - obv_ma20) / (obv_ma20.abs() + 1)

        high_low = data[StockCol.HIGH] - data[StockCol.LOW]
        high_close = (data[StockCol.HIGH] - data[StockCol.CLOSE].shift()).abs()
        low_close = (data[StockCol.LOW] - data[StockCol.CLOSE].shift()).abs()

        # 算出每日真實波幅 (True Range)
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # 計算 14 日平均真實波幅 (ATR)
        atr_14 = true_range.rolling(window=14).mean()
        # 正規化為百分比，這樣 10 元和 1000 元的股票才能在 AI 腦中公平比較
        data[FeatureCol.ATR_RATIO] = atr_14 / (data[ai_vision_col] + 1e-9)

        # 用 10 日均線的斜率絕對值，加上 20 日均線的斜率絕對值
        ma10 = data[ai_vision_col].rolling(window=10).mean()
        slope_10 = (ma10 - ma10.shift(5)) / (ma10.shift(5) + 1e-9)
        slope_20 = (ma_m - ma_m.shift(10)) / (ma_m.shift(10) + 1e-9)
        # 強度 = 短期與中期趨勢動能的疊加絕對值 (不在乎多空，只在乎盤勢黏不黏)
        data[FeatureCol.TREND_STRENGTH] = slope_10.abs() + slope_20.abs()

        # 價格與成交量動能
        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()
        data[FeatureCol.CLOSE_CHANGE] = data[ai_vision_col].pct_change()
        data[FeatureCol.RETURN_5D] = data[ai_vision_col].pct_change(periods=5)

        # 更穩健的 K 線微結構特徵計算 (防止一字鎖死盤的浮點數毒害)
        max_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].max(axis=1)
        min_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].min(axis=1)

        # 使用 clip 確保分母至少有合理的跳動單位 (假設為 0.01)，取代單純的 1e-9
        price_range = (data[StockCol.HIGH] - data[StockCol.LOW]).clip(lower=0.01)

        # 上影線比例、下影線比例、實體 K 線比例
        data[FeatureCol.K_UPPER] = (data[StockCol.HIGH] - max_open_close) / price_range
        data[FeatureCol.K_LOWER] = (min_open_close - data[StockCol.LOW]) / price_range
        data[FeatureCol.K_BODY] = (data[StockCol.CLOSE] - data[StockCol.OPEN]) / price_range

        # 當日買盤力道 (收盤價在當日震幅的相對位置，0.5 為中性)
        data[FeatureCol.BUY_POWER] = (data[StockCol.CLOSE] - data[StockCol.LOW]) / price_range

        # 大盤前綴在 DataManager 裡是被 replace 掉 '^' 符號的 (TWII_)
        twii_prefix = MacroTicker.TWII.value.replace('^', '') + "_"
        twii_close_col = f"{twii_prefix}{StockCol.CLOSE.value}" if hasattr(StockCol.CLOSE, 'value') else f"{twii_prefix}close"

        if twii_close_col in data.columns:
            stock_ma5 = data[ai_vision_col].rolling(window=5).mean()
            stock_ma20 = ma_m
            stock_momentum = (stock_ma5 - stock_ma20) / (stock_ma20 + 1e-9)

            twii_ma5 = data[twii_close_col].rolling(window=5).mean()
            twii_ma20 = data[twii_close_col].rolling(window=20).mean()
            twii_momentum = (twii_ma5 - twii_ma20) / (twii_ma20 + 1e-9)

            # 相對強弱 = 個股均線動能 - 大盤均線動能
            data[FeatureCol.RS_5D] = stock_momentum - twii_momentum
        else:
            # 防呆：如果找不到大盤資料，填 0
            data[FeatureCol.RS_5D] = 0.0

        return data

    @staticmethod
    def _create_labels(df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """
        建立預測目標 (y)：未來 N 天的收盤價是否大於今天的收盤價？
        1 代表看漲 (Up)，0 代表看跌或盤整 (Down)
        """
        if df.empty: return df

        data = df.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        adj_factor = data[ai_vision_col] / (data[StockCol.CLOSE] + 1e-9)
        adj_high = data[StockCol.HIGH] * adj_factor

        future_high_max = adj_high.rolling(window=lookahead, min_periods=1).max().shift(-lookahead)
        target_condition = future_high_max > (data[ai_vision_col] * 1.025)

        data[FeatureCol.TARGET] = target_condition.astype('Int64')
        data.loc[future_high_max.isna(), FeatureCol.TARGET] = pd.NA

        return data