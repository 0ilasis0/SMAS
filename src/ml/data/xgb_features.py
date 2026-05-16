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

        features = FeatureCol.get_xgb_features()

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

        # 1. 趨勢與乖離
        ma_w = data[StockCol.ADJ_CLOSE].rolling(window=self.params.MA_WEEK).mean()
        ma_m = data[StockCol.ADJ_CLOSE].rolling(window=self.params.MA_MONTH).mean()
        ma_q = data[StockCol.ADJ_CLOSE].rolling(window=self.params.MA_QUARTER).mean()
        ma_y = data[StockCol.ADJ_CLOSE].rolling(window=self.params.MA_YEAR).mean()

        data[FeatureCol.BIAS_WEEK] = (data[StockCol.ADJ_CLOSE] - ma_w) / (ma_w + 1e-9)
        data[FeatureCol.BIAS_MONTH] = (data[StockCol.ADJ_CLOSE] - ma_m) / (ma_m + 1e-9)
        data[FeatureCol.BIAS_QUARTER] = (data[StockCol.ADJ_CLOSE] - ma_q) / (ma_q + 1e-9)
        data[FeatureCol.BIAS_YEAR] = (data[StockCol.ADJ_CLOSE] - ma_y) / (ma_y + 1e-9)

        # 2. 動能與震盪
        ema_fast = data[StockCol.ADJ_CLOSE].ewm(span=self.params.MACD_FAST, adjust=False).mean()
        ema_slow = data[StockCol.ADJ_CLOSE].ewm(span=self.params.MACD_SLOW, adjust=False).mean()
        data[FeatureCol.MACD] = (ema_fast - ema_slow) / (data[StockCol.ADJ_CLOSE] + 1e-9) * 100
        data[FeatureCol.MACD_SIGNAL] = data[FeatureCol.MACD].ewm(span=self.params.MACD_SIGNAL, adjust=False).mean()

        # KD 指標
        kd_rsv_w = self.params.KD_RSV_WINDOW
        kd_com = self.params.KD_SMOOTH - 1
        low_kd = data[StockCol.LOW].rolling(window=kd_rsv_w).min()
        high_kd = data[StockCol.HIGH].rolling(window=kd_rsv_w).max()
        rsv = (data[StockCol.CLOSE] - low_kd) / (high_kd - low_kd + 1e-9) * 100
        data[FeatureCol.KD_K] = rsv.ewm(com=kd_com, adjust=False).mean()
        data[FeatureCol.KD_D] = data[FeatureCol.KD_K].ewm(com=kd_com, adjust=False).mean()
        data[FeatureCol.KD_CROSS] = data[FeatureCol.KD_K] - data[FeatureCol.KD_D]

        # 3. 量能與波動
        direction = np.sign(data[StockCol.ADJ_CLOSE].diff()).fillna(0)
        raw_obv = (direction * data[StockCol.VOLUME]).cumsum()
        obv_ma20 = raw_obv.rolling(window=20).mean()
        data[FeatureCol.OBV] = (raw_obv - obv_ma20) / (obv_ma20.abs() + 1)
        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()

        high_low = data[StockCol.HIGH] - data[StockCol.LOW]
        high_close = (data[StockCol.HIGH] - data[StockCol.CLOSE].shift()).abs()
        low_close = (data[StockCol.LOW] - data[StockCol.CLOSE].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data[FeatureCol.ATR_RATIO] = true_range.rolling(window=14).mean() / (data[StockCol.ADJ_CLOSE] + 1e-9)

        # 4. K 線幾何與跳空缺口
        prev_close = data[StockCol.CLOSE].shift(1)
        data[FeatureCol.GAP_RATIO] = (data[StockCol.OPEN] - prev_close) / (prev_close + 1e-9)
        data[FeatureCol.CLOSE_CHANGE] = data[StockCol.ADJ_CLOSE].pct_change()

        max_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].max(axis=1)
        min_open_close = data[[StockCol.OPEN, StockCol.CLOSE]].min(axis=1)
        price_range = (data[StockCol.HIGH] - data[StockCol.LOW]).clip(lower=0.01)

        data[FeatureCol.K_UPPER] = (data[StockCol.HIGH] - max_open_close) / price_range
        data[FeatureCol.K_LOWER] = (min_open_close - data[StockCol.LOW]) / price_range

        # 5. 大盤相對強弱
        twii_prefix = MacroTicker.TWII.value.replace('^', '') + "_"
        twii_close_col = f"{twii_prefix}{StockCol.CLOSE.value}" if hasattr(StockCol.CLOSE, 'value') else f"{twii_prefix}close"

        if twii_close_col in data.columns:
            stock_ma5 = data[StockCol.ADJ_CLOSE].rolling(window=5).mean()
            stock_ma10 = data[StockCol.ADJ_CLOSE].rolling(window=10).mean()
            twii_ma5 = data[twii_close_col].rolling(window=5).mean()
            twii_ma10 = data[twii_close_col].rolling(window=10).mean()
            twii_ma20 = data[twii_close_col].rolling(window=20).mean()

            stock_momentum_5 = (stock_ma5 - ma_m) / (ma_m + 1e-9)
            twii_momentum_5 = (twii_ma5 - twii_ma20) / (twii_ma20 + 1e-9)
            data[FeatureCol.RS_5D] = stock_momentum_5 - twii_momentum_5

            stock_momentum_10 = (stock_ma10 - ma_m) / (ma_m + 1e-9)
            twii_momentum_10 = (twii_ma10 - twii_ma20) / (twii_ma20 + 1e-9)
            data[FeatureCol.RS_10D] = stock_momentum_10 - twii_momentum_10
        else:
            data[FeatureCol.RS_5D] = 0.0
            data[FeatureCol.RS_10D] = 0.0

        return data

    def _create_labels(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        if df.empty: return df
        data = df.copy()

        adj_factor = data[StockCol.ADJ_CLOSE] / (data[StockCol.CLOSE] + 1e-9)
        adj_high = data[StockCol.HIGH] * adj_factor
        adj_low = data[StockCol.LOW] * adj_factor

        # 1. 計算真實波幅 ATR
        high_low = adj_high - adj_low
        high_close = (adj_high - data[StockCol.ADJ_CLOSE].shift()).abs()
        low_close = (adj_low - data[StockCol.ADJ_CLOSE].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.entry_criteria.ATR_LOOKBACK).mean()

        # 2. 動態設定目標與停損價位
        target_atr = self.entry_criteria.PROFIT_TARGET_ATR
        stop_atr = self.entry_criteria.STOP_LOSS_ATR

        target_profit_price = data[StockCol.ADJ_CLOSE] + (atr * target_atr)
        stop_loss_price = data[StockCol.ADJ_CLOSE] - (atr * stop_atr)

        hit_target_day = pd.Series(np.inf, index=data.index)
        hit_stop_day = pd.Series(np.inf, index=data.index)

        # 3. 實戰時間迴圈模擬器 (尋找先碰到誰)
        for i in range(1, lookahead + 1):
            future_high = adj_high.shift(-i)
            future_low = adj_low.shift(-i)

            target_mask = (future_high >= target_profit_price) & (hit_target_day == np.inf)
            hit_target_day.loc[target_mask] = i

            stop_mask = (future_low <= stop_loss_price) & (hit_stop_day == np.inf)
            hit_stop_day.loc[stop_mask] = i

        # 4. ➔ 貫徹特種部隊標籤法：只有先碰到目標、且未碰到停損才算成功突破
        target_condition = (hit_target_day != np.inf) & (hit_target_day < hit_stop_day)

        # 轉換為 0 或 1 的整數標籤
        data[FeatureCol.TARGET] = target_condition.astype(int)

        # 過濾未來資料尚不足夠的尾端天數
        valid_future_mask = data[StockCol.ADJ_CLOSE].shift(-lookahead).notna()
        data.loc[~valid_future_mask, FeatureCol.TARGET] = pd.NA

        return data