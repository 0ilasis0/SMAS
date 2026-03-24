import numpy as np
import pandas as pd
from data.variable import StockCol
from debug import dbg
from ml.params import FeatureCol
from sklearn.preprocessing import MinMaxScaler


class DLFeatureEngine:
    """
    專為時序深度學習模型 (TCN/LSTM) 設計的特徵工程。
    負責特徵縮放 (MinMaxScaler) 與產生 3D 滑動視窗矩陣 (Sliding Window)。
    """
    def __init__(self, time_steps: int = 60, lookahead: int = 20):
        self.time_steps = time_steps  # 模型要回看過去幾根 K 線 (例如 60 天)
        self.lookahead = lookahead    # 預測未來幾天後的漲跌

    def process_pipeline(self, df: pd.DataFrame):
        """
        執行完整的 DL 特徵管線。
        回傳: X (3D Numpy Array), y (1D Numpy Array), scaler (用來在線上推論時縮放新資料)
        """
        dbg.log("開始建立 Deep Learning 時序特徵矩陣 (Sliding Window)...")

        if df.empty or len(df) <= self.time_steps + self.lookahead:
            dbg.war("資料量不足以生成時序視窗。")
            return None, None, None

        data = df.copy()

        # 1. 建立標籤 (與 XGBoost 邏輯完全一致，確保雙軌預測目標相同)
        future_close = data[StockCol.CLOSE].shift(self.lookahead * (-1))
        data[FeatureCol.TARGET] = (future_close > data[StockCol.CLOSE]).astype('Int64')
        data.loc[future_close.isna(), FeatureCol.TARGET] = pd.NA

        # 2. 移除未來盲區產生的 NaN
        data = data.dropna(subset=[FeatureCol.TARGET])

        # 3. 選取要餵給神經網路的原始特徵 (通常神經網路自己會學指標，所以我們只餵原始 OHLCV)
        features = [StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.CLOSE, StockCol.VOLUME]

        # 4. 特徵正規化 (Scaling 到 0 ~ 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(data[features])

        # 5. 建立滑動視窗 (Sliding Window)
        X, y = [], []
        targets = data[FeatureCol.TARGET].values

        # 迴圈邏輯：從第 time_steps-1 天開始，才能往前抓滿 time_steps 根 K 線
        for i in range(self.time_steps - 1, len(scaled_features)):
            # X: 擷取 [今天 - 59天 : 今天 + 1] (總共 60 根 K 線)
            window_data = scaled_features[i - self.time_steps + 1 : i + 1]
            X.append(window_data)

            # y: 當天的標籤 (未來 lookahead 天是否上漲)
            y.append(targets[i])

        X = np.array(X)
        y = np.array(y).astype(int)

        dbg.log(f"時序矩陣建立完成！ X 形狀: {X.shape}, y 形狀: {y.shape}")
        return X, y, scaler