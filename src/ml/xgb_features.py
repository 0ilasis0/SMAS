import pandas as pd
from data.variable import StockCol
from debug import dbg
from ml.params import FeatureCol, IndicatorParams


class XGBFeatureEngine:
    """
    以 XGBoost 設計的特徵工程。
    負責計算技術指標 (MA, RSI, MACD) 與生成預測標籤 (Target)。
    """
    def __init__(self, params: IndicatorParams = IndicatorParams()):
        self.params = params

    def process_pipeline(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """執行完整的 XGBoost 特徵管線"""
        dbg.log("開始計算 XGBoost 技術特徵與標籤...")

        df_features = self._create_daily_features(df)
        df_labeled = self._create_labels(df_features, lookahead)

        initial_len = len(df_labeled)
        df_clean = df_labeled.dropna()
        final_len = len(df_clean)

        dbg.log(f"特徵工程完成。移除了 {initial_len - final_len} 筆含 NaN 的無效資料，剩餘 {final_len} 筆可用樣本。")
        return df_clean

    def _create_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算波段 (日 K) 特徵"""
        if df.empty: return df

        data = df.copy()

        #  移動平均線
        data[FeatureCol.MA_WEEK] = data[StockCol.CLOSE].rolling(window=self.params.MA_WEEK).mean()
        data[FeatureCol.MA_MONTH] = data[StockCol.CLOSE].rolling(window=self.params.MA_MONTH).mean()
        data[FeatureCol.MA_QUARTER] = data[StockCol.CLOSE].rolling(window=self.params.MA_QUARTER).mean()
        data[FeatureCol.MA_YEAR] = data[StockCol.CLOSE].rolling(window=self.params.MA_YEAR).mean()

        # RSI (相對強弱指標)
        delta = data[StockCol.CLOSE].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.RSI_PERIOD).mean()
        rs = gain / (loss + 1e-9)
        data[FeatureCol.RSI] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = data[StockCol.CLOSE].ewm(span=self.params.MACD_FAST, adjust=False).mean()
        ema_slow = data[StockCol.CLOSE].ewm(span=self.params.MACD_SLOW, adjust=False).mean()
        data[FeatureCol.MACD] = ema_fast - ema_slow
        data[FeatureCol.MACD_SIGNAL] = data[FeatureCol.MACD].ewm(span=self.params.MACD_SIGNAL, adjust=False).mean()

        # 價格與成交量動能
        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()
        data[FeatureCol.CLOSE_CHANGE] = data[StockCol.CLOSE].pct_change()

        return data

    @staticmethod
    def _create_labels(df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """
        建立預測目標 (y)：未來 N 天的收盤價是否大於今天的收盤價？
        1 代表看漲 (Up)，0 代表看跌或盤整 (Down)
        """
        if df.empty: return df

        data = df.copy()

        # 將未來第 N 天的收盤價往回拉到今天的 Row
        future_close = data[StockCol.CLOSE].shift(-lookahead)

        # 先轉為支援缺失值的整數型態，再將確實沒有未來資料的列強制設為 NaN
        data[FeatureCol.TARGET] = (future_close > data[StockCol.CLOSE]).astype('Int64')
        data.loc[future_close.isna(), FeatureCol.TARGET] = pd.NA

        return data
