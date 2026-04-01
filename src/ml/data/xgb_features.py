# ml/xgb_features.py
import numpy as np
import pandas as pd

from data.const import StockCol
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

        # 移動平均線乖離率
        data[FeatureCol.BIAS_WEEK] = (data[ai_vision_col] - ma_w) / ma_w
        data[FeatureCol.BIAS_MONTH] = (data[ai_vision_col] - ma_m) / ma_m
        data[FeatureCol.BIAS_QUARTER] = (data[ai_vision_col] - ma_q) / ma_q
        data[FeatureCol.BIAS_YEAR] = (data[ai_vision_col] - ma_y) / ma_y

        # RSI (相對強弱指標)
        delta = data[ai_vision_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.RSI_PERIOD).mean()
        rs = gain / (loss + 1e-9)
        data[FeatureCol.RSI] = 100 - (100 / (1 + rs))

        # MACD (轉化為百分比 PPO)
        ema_fast = data[ai_vision_col].ewm(span=self.params.MACD_FAST, adjust=False).mean()
        ema_slow = data[ai_vision_col].ewm(span=self.params.MACD_SLOW, adjust=False).mean()
        data[FeatureCol.MACD] = (ema_fast - ema_slow) / data[ai_vision_col] * 100
        data[FeatureCol.MACD_SIGNAL] = data[FeatureCol.MACD].ewm(span=self.params.MACD_SIGNAL, adjust=False).mean()

        # 布林通道寬度 (BB Width) - 抓波動率壓縮
        rolling_std = data[ai_vision_col].rolling(window=self.params.MA_MONTH).std()
        data[FeatureCol.BB_WIDTH] = (rolling_std * 2) / ma_m

        # 價格與成交量動能
        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()
        data[FeatureCol.CLOSE_CHANGE] = data[ai_vision_col].pct_change()
        data[FeatureCol.RETURN_5D] = data[ai_vision_col].pct_change(periods=5)

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
