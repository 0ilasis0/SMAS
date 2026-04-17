import numpy as np
import pandas as pd

from data.const import MacroTicker, StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.params import EntryQualityCriteria, IndicatorParams


class XGBFeatureEngine:
    """
    以 XGBoost 設計的特徵工程。
    負責計算技術指標 (MA, RSI, MACD, OBV, ATR) 與生成預測標籤 (Target)。
    """
    def __init__(self, params: IndicatorParams = IndicatorParams(), entry_criteria: EntryQualityCriteria = EntryQualityCriteria()):
        self.params = params
        self.entry_criteria = entry_criteria

    def process_pipeline(self, df: pd.DataFrame, lookahead: int, is_training: bool = True) -> pd.DataFrame:
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

        rolling_std = data[ai_vision_col].rolling(window=self.params.MA_MONTH).std()
        data[FeatureCol.BB_WIDTH] = (rolling_std * 2) / (ma_m + 1e-9)

        price_diff = data[ai_vision_col].diff()
        direction = np.sign(price_diff)
        direction = direction.fillna(0)
        raw_obv = (direction * data[StockCol.VOLUME]).cumsum()
        obv_ma20 = raw_obv.rolling(window=20).mean()
        data[FeatureCol.OBV] = (raw_obv - obv_ma20) / (obv_ma20.abs() + 1)

        high_low = data[StockCol.HIGH] - data[StockCol.LOW]
        high_close = (data[StockCol.HIGH] - data[StockCol.CLOSE].shift()).abs()
        low_close = (data[StockCol.LOW] - data[StockCol.CLOSE].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = true_range.rolling(window=14).mean()
        data[FeatureCol.ATR_RATIO] = atr_14 / (data[ai_vision_col] + 1e-9)

        ma10 = data[ai_vision_col].rolling(window=10).mean()
        slope_10 = (ma10 - ma10.shift(5)) / (ma10.shift(5) + 1e-9)
        slope_20 = (ma_m - ma_m.shift(10)) / (ma_m.shift(10) + 1e-9)
        data[FeatureCol.TREND_STRENGTH] = slope_10.abs() + slope_20.abs()

        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()
        data[FeatureCol.CLOSE_CHANGE] = data[ai_vision_col].pct_change()
        data[FeatureCol.RETURN_5D] = data[ai_vision_col].pct_change(periods=5)

        max_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].max(axis=1)
        min_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].min(axis=1)
        price_range = (data[StockCol.HIGH] - data[StockCol.LOW]).clip(lower=0.01)

        data[FeatureCol.K_UPPER] = (data[StockCol.HIGH] - max_open_close) / price_range
        data[FeatureCol.K_LOWER] = (min_open_close - data[StockCol.LOW]) / price_range
        data[FeatureCol.K_BODY] = (data[StockCol.CLOSE] - data[StockCol.OPEN]) / price_range
        data[FeatureCol.BUY_POWER] = (data[StockCol.CLOSE] - data[StockCol.LOW]) / price_range

        twii_prefix = MacroTicker.TWII.value.replace('^', '') + "_"
        twii_close_col = f"{twii_prefix}{StockCol.CLOSE.value}" if hasattr(StockCol.CLOSE, 'value') else f"{twii_prefix}close"

        if twii_close_col in data.columns:
            stock_ma5 = data[ai_vision_col].rolling(window=5).mean()
            stock_ma20 = ma_m
            stock_momentum = (stock_ma5 - stock_ma20) / (stock_ma20 + 1e-9)

            twii_ma5 = data[twii_close_col].rolling(window=5).mean()
            twii_ma20 = data[twii_close_col].rolling(window=20).mean()
            twii_momentum = (twii_ma5 - twii_ma20) / (twii_ma20 + 1e-9)

            data[FeatureCol.RS_5D] = stock_momentum - twii_momentum
        else:
            data[FeatureCol.RS_5D] = 0.0

        return data

    def _create_labels(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        if df.empty: return df
        data = df.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        # 1. 嚴謹處理除權息，確保高低價判定不會被權值缺口干擾
        adj_factor = data[ai_vision_col] / (data[StockCol.CLOSE] + 1e-9)
        adj_high = data[StockCol.HIGH] * adj_factor
        adj_low = data[StockCol.LOW] * adj_factor

        # 2. 計算真實波幅 ATR
        high_low = adj_high - adj_low
        high_close = (adj_high - data[ai_vision_col].shift()).abs()
        low_close = (adj_low - data[ai_vision_col].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.entry_criteria.ATR_LOOKBACK).mean()

        # 3. 動態設定目標與停損價位
        target_profit_price = data[ai_vision_col] + (atr * self.entry_criteria.PROFIT_TARGET_ATR)
        stop_loss_price = data[ai_vision_col] - (atr * self.entry_criteria.STOP_LOSS_ATR)

        # 4. 初始化觸發紀錄 (預設 inf 代表沒碰到)
        hit_target_day = pd.Series(np.inf, index=data.index)
        hit_stop_day = pd.Series(np.inf, index=data.index)

        # 5. 實戰時間迴圈模擬器 (尋找先碰到誰)
        for i in range(1, lookahead + 1):
            future_high = adj_high.shift(-i)
            future_low = adj_low.shift(-i)

            target_mask = (future_high >= target_profit_price) & (hit_target_day == np.inf)
            hit_target_day.loc[target_mask] = i

            stop_mask = (future_low <= stop_loss_price) & (hit_stop_day == np.inf)
            hit_stop_day.loc[stop_mask] = i

        # 6. 最終標籤判定：有碰到目標，且比停損早碰到
        target_condition = (hit_target_day != np.inf) & (hit_target_day < hit_stop_day)

        # 過濾未來資料尚不足夠的尾端天數
        valid_future_mask = data[ai_vision_col].shift(-lookahead).notna()

        data[FeatureCol.TARGET] = target_condition.astype('Int64')
        data.loc[~valid_future_mask, FeatureCol.TARGET] = pd.NA

        return data