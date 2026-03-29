import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.const import StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.params import DLHyperParams


class DLFeatureEngine:
    """
    專為時序深度學習模型 (CNN/LSTM) 設計的特徵工程 (純動能視角)。
    僅負責生成原始特徵與 3D 滑動視窗，正規化 (Scaling) 交由 Trainer 在 CV 迴圈內處理以防止未來資料外洩。
    """
    def __init__(
            self,
            lookahead: int,
            time_steps: int = DLHyperParams.TIME_STEPS
        ):
        self.lookahead = lookahead
        self.time_steps = time_steps
        self.max_warmup = 19

    def process_pipeline(self, df: pd.DataFrame, is_training: bool = True):
        dbg.log("開始建立 Deep Learning 原始時序特徵矩陣 (Sliding Window)...")

        min_required_len = self.time_steps + self.max_warmup
        if is_training:
            min_required_len += self.lookahead

        if df.empty or len(df) <= min_required_len:
            dbg.war(f"資料量不足。需要 {min_required_len} 筆 (含暖機期)，目前僅有 {len(df)} 筆。")
            return None, None, None

        data = df.copy().ffill()

        new_features = {}
        dl_features = []

        # 基礎 K 線變化 (對數報酬率)
        for col in StockCol.get_ohlcv():
            feat_name = f"{col}_log_chg"
            if col == StockCol.VOLUME:
                new_features[feat_name] = np.log1p(data[col]) - np.log1p(data[col].shift(1))
            else:
                new_features[feat_name] = np.log(data[col] / (data[col].shift(1) + 1e-9))
            dl_features.append(feat_name)

        ma_w = data[StockCol.CLOSE].rolling(window=5).mean()
        ma_m = data[StockCol.CLOSE].rolling(window=20).mean()
        rolling_std = data[StockCol.CLOSE].rolling(window=20).std()

        new_features[FeatureCol.BIAS_WEEK] = (data[StockCol.CLOSE] - ma_w) / (ma_w + 1e-9)
        new_features[FeatureCol.BIAS_MONTH] = (data[StockCol.CLOSE] - ma_m) / (ma_m + 1e-9)
        new_features[FeatureCol.BB_WIDTH] = (rolling_std * 2) / (ma_m + 1e-9)

        dl_features.extend([FeatureCol.BIAS_WEEK, FeatureCol.BIAS_MONTH, FeatureCol.BB_WIDTH])

        data = data.assign(**new_features)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=dl_features)

        if len(data) < self.time_steps:
            dbg.war("扣除暖機期後，資料量不足以建立滑動視窗。")
            return None, None, None

        # 直接使用未縮放的原始特徵建立滑動視窗
        raw_features = data[dl_features].values

        # 建立滑動視窗 (batch, features, time_steps)
        X = sliding_window_view(raw_features, window_shape=self.time_steps, axis=0)
        # 轉置給 PyTorch LSTM (batch, time_steps, features)
        X = np.transpose(X, (0, 2, 1))

        aligned_index = data.index[self.time_steps - 1:]

        if is_training:
            future_high_max = data[StockCol.HIGH].rolling(window=self.lookahead, min_periods=1).max().shift(-self.lookahead)
            target_condition = future_high_max > (data[StockCol.CLOSE] * 1.025)

            y_all = target_condition.astype(int).values
            y = y_all[self.time_steps - 1:]

            future_isna = future_high_max.isna().values[self.time_steps - 1:]
            valid_mask = ~future_isna

            X = X[valid_mask]
            y = np.array(y[valid_mask]).astype(int)
            valid_index = aligned_index[valid_mask]
        else:
            y = None
            valid_index = aligned_index

        y_shape_str = str(y.shape) if y is not None else "None"
        dbg.log(f"時序矩陣建立完成！ X 原始形狀: {X.shape}, y 形狀: {y_shape_str}")

        # 回傳原始矩陣，交由 Trainer 處理縮放
        return X, y, valid_index
