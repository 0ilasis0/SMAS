import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler

from data.const import StockCol
from debug import dbg
from ml.params import DLHyperParams


class DLFeatureEngine:
    """
    專為時序深度學習模型 (TCN/LSTM) 設計的特徵工程 (純動能視角)。
    負責特徵縮放 (RobustScaler) 與產生 3D 滑動視窗矩陣 (Sliding Window)。
    :para: lookahead -> 預測未來幾天後的漲跌
    :para: time_steps -> 模型要回看過去幾根(天) K 線
    """
    def __init__(
            self,
            lookahead: int,
            time_steps: int = DLHyperParams.TIME_STEPS
        ):
        self.lookahead = lookahead
        self.time_steps = time_steps

    def process_pipeline(self, df: pd.DataFrame, scaler: RobustScaler | None = None):
        """
        執行 DL 特徵管線。
        :param df: 原始 DataFrame
        :param scaler: 若傳入已訓練好的 Scaler，則進入「推論/測試模式」；若不傳入，則進入「訓練模式」。
        :return: X (3D Numpy Array), y (1D Numpy Array), scaler (用來在線上推論時縮放新資料), valid_index(對應的正確日期 Index)
        """
        dbg.log("開始建立 Deep Learning 時序特徵矩陣 (Sliding Window)...")

        is_training = scaler is None
        min_required_len = self.time_steps + self.lookahead if is_training else self.time_steps

        if df.empty or len(df) <= min_required_len:
            dbg.war(f"資料量不足。需要 {min_required_len} 筆，目前僅有 {len(df)} 筆。")
            return None, None, None, None

        data = df.copy()

        # 選取要餵給神經網路的原始特徵
        dl_features = []
        for col in StockCol.get_ohlcv():
            feat_name = f"{col}_log_chg"
            data[feat_name] = np.log(data[col] / data[col].shift(1))
            dl_features.append(feat_name)

        # 處理極端值與 0 填補
        data[dl_features] = data[dl_features].replace([np.inf, -np.inf], np.nan)
        data[dl_features] = data[dl_features].fillna(0)

        # 特徵正規化 (Scaling 到 0 ~ 1)
        if is_training:
            dbg.log("訓練模式：重新 Fit Scaler")
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(data[dl_features])
        else:
            dbg.log("推論模式：使用既有 Scaler 進行 Transform")
            scaled_features = scaler.transform(data[dl_features])

        # 建立滑動視窗
        X = sliding_window_view(scaled_features, window_shape=self.time_steps, axis=0)
        X = np.transpose(X, (0, 2, 1))

        # 對齊索引
        aligned_index = data.index[self.time_steps - 1:]

        # 處理標籤 (Target) 與切分
        if is_training:
            future_close = data[StockCol.CLOSE].shift(-self.lookahead)

            # 直接算出所有的 0 和 1 (忽略 NaN，反正最後會切掉)
            y_all = (future_close > data[StockCol.CLOSE]).astype(int).values
            y = y_all[self.time_steps - 1:]

            # 直接用 future_close 是否為 NaN 來做 Mask
            future_isna = future_close.isna().values[self.time_steps - 1:]
            valid_mask = ~future_isna

            X = X[valid_mask]
            y = np.array(y[valid_mask]).astype(int)
            valid_index = aligned_index[valid_mask]
        else:
            y = None
            valid_index = aligned_index

        y_shape_str = str(y.shape) if y is not None else "None"
        dbg.log(f"時序矩陣建立完成！ X 形狀: {X.shape}, y 形狀: {y_shape_str}")

        return X, y, scaler, valid_index
