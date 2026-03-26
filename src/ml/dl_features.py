import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.variable import StockCol
from debug import dbg
from ml.params import FeatureCol, IndicatorParams


class DLFeatureEngine:
    """
    專為時序深度學習模型 (TCN/LSTM) 設計的特徵工程。
    負責特徵縮放 (MinMaxScaler) 與產生 3D 滑動視窗矩陣 (Sliding Window)。
    """
    def __init__(
            self,
            time_steps: int = IndicatorParams.MA_QUARTER,
            lookahead: int = IndicatorParams.MA_MONTH
        ):
        # 模型要回看過去幾根(天) K 線
        self.time_steps = time_steps
        # 預測未來幾天後的漲跌
        self.lookahead = lookahead

    def process_pipeline(self, df: pd.DataFrame, scaler: MinMaxScaler | None = None):
        """
        執行 DL 特徵管線。
        :param df: 原始 DataFrame
        :param scaler: 若傳入已訓練好的 Scaler，則進入「推論/測試模式」；若不傳入，則進入「訓練模式」。
        :return: X (3D Numpy Array), y (1D Numpy Array), scaler (用來在線上推論時縮放新資料)
        """
        dbg.log("開始建立 Deep Learning 時序特徵矩陣 (Sliding Window)...")

        is_training = scaler is None
        min_required_len = self.time_steps + self.lookahead if is_training else self.time_steps

        if df.empty or len(df) <= min_required_len:
            dbg.war(f"資料量不足。需要 {min_required_len} 筆，目前僅有 {len(df)} 筆。")
            return None, None, None

        data = df.copy()

        # 建立標籤
        future_close = data[StockCol.CLOSE].shift(-self.lookahead)
        data[FeatureCol.TARGET] = (future_close > data[StockCol.CLOSE]).astype('Int64')
        data.loc[future_close.isna(), FeatureCol.TARGET] = pd.NA

        # 選取要餵給神經網路的原始特徵
        features = FeatureCol.get_features()
        data = data.replace([np.inf, -np.inf], np.nan)

        # 特徵正規化 (Scaling 到 0 ~ 1)
        if is_training:
            # 產生全新的 Scaler，並從訓練資料中學習 (fit) 最大最小值
            dbg.log("訓練模式：重新 Fit Scaler")
            data = data.dropna(subset=features + [FeatureCol.TARGET])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data[features])
        else:
            # 嚴格禁止使用 fit，只能使用訓練集傳過來的 Scaler 進行轉換
            dbg.log("推論模式：使用既有 Scaler 進行 Transform")
            data = data.dropna(subset=features)
            scaled_features = scaler.transform(data[features])

        # 建立滑動視窗
        X = sliding_window_view(scaled_features, window_shape=self.time_steps, axis=0)
        X = np.transpose(X, (0, 2, 1))

        if is_training:
            # 回傳完整的 X 矩陣與對應的標籤 y
            targets = data[FeatureCol.TARGET].values
            y = targets[self.time_steps - 1:]
            y = np.array(y).astype(int)
        else:
            # 實戰中，我們通常只需要拿「最後一個 Window (即最新資料)」去預測未來
            X = X[-1:]
            y = None   # 推論時沒有標準答案

        y_shape_str = str(y.shape) if y is not None else "None"
        dbg.log(f"時序矩陣建立完成！ X 形狀: {X.shape}, y 形狀: {y_shape_str}")
        return X, y, scaler
