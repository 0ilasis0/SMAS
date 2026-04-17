import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.const import StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.params import DLHyperParams, EntryQualityCriteria, IndicatorParams


class DLFeatureEngine:
    """
    專為時序深度學習模型 (CNN/LSTM) 設計的特徵工程 (純動能視角)。
    僅負責生成原始特徵與 3D 滑動視窗，正規化 (Scaling) 交由 Trainer 在 CV 迴圈內處理以防止未來資料外洩。
    """
    def __init__(
            self,
            lookahead: int,
            time_steps: int = DLHyperParams.TIME_STEPS,
            entry_criteria: EntryQualityCriteria = EntryQualityCriteria()
        ):
        self.lookahead = lookahead
        self.time_steps = time_steps
        self.entry_criteria = entry_criteria
        self.max_warmup = 19

    def process_pipeline(self, df: pd.DataFrame, is_training: bool = True):
        if df is None or df.empty:
            dbg.war("⚠️ [DLFeatureEngine] 輸入 DataFrame 為空，無法計算特徵。")
            return None, None, None

        dbg.log("開始建立 Deep Learning 原始時序特徵矩陣 (Sliding Window)...")

        min_required_len = self.time_steps + self.max_warmup
        if is_training:
            min_required_len += self.lookahead

        if df.empty or len(df) <= min_required_len:
            dbg.war(f"資料量不足。需要 {min_required_len} 筆 (含暖機期)，目前僅有 {len(df)} 筆。")
            return None, None, None

        data = df.copy().ffill()

        adj_factor = data[StockCol.ADJ_CLOSE] / (data[StockCol.CLOSE] + 1e-9)
        target_cols = [StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.VOLUME, StockCol.ADJ_CLOSE]

        new_features = {}
        dl_features = []

        for col in target_cols:
            feat_name = f"{col.value}_log_chg"
            if col == StockCol.VOLUME:
                new_features[feat_name] = np.log1p(data[col]) - np.log1p(data[col].shift(1))
            else:
                adj_price = data[col] * adj_factor if col != StockCol.ADJ_CLOSE else data[col]
                prev_adj_price = data[col].shift(1) * adj_factor.shift(1) if col != StockCol.ADJ_CLOSE else data[col].shift(1)
                new_features[feat_name] = np.log(adj_price / (prev_adj_price + 1e-9))
            dl_features.append(feat_name)

        ai_vision_col = str(StockCol.ADJ_CLOSE)

        ma_w = data[ai_vision_col].rolling(window=IndicatorParams.MA_WEEK).mean()
        ma_m = data[ai_vision_col].rolling(window=IndicatorParams.MA_MONTH).mean()
        rolling_std = data[ai_vision_col].rolling(window=IndicatorParams.MA_MONTH).std()

        new_features[FeatureCol.BIAS_WEEK] = (data[ai_vision_col] - ma_w) / (ma_w + 1e-9)
        new_features[FeatureCol.BIAS_MONTH] = (data[ai_vision_col] - ma_m) / (ma_m + 1e-9)
        new_features[FeatureCol.BB_WIDTH] = (rolling_std * 2) / (ma_m + 1e-9)

        dl_features.extend([FeatureCol.BIAS_WEEK, FeatureCol.BIAS_MONTH, FeatureCol.BB_WIDTH])

        data = data.assign(**new_features)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=dl_features)

        if len(data) < self.time_steps:
            dbg.war("扣除暖機期後，資料量不足以建立滑動視窗。")
            return None, None, None

        raw_features = data[dl_features].values
        X = sliding_window_view(raw_features, window_shape=self.time_steps, axis=0)
        X = np.transpose(X, (0, 2, 1))

        aligned_index = data.index[self.time_steps - 1:]

        if is_training:
            # 1. 還原權值避免高低點判斷失真
            current_adj_factor = data[StockCol.ADJ_CLOSE] / (data[StockCol.CLOSE] + 1e-9)
            adj_high = data[StockCol.HIGH] * current_adj_factor
            adj_low = data[StockCol.LOW] * current_adj_factor

            # 2. 計算真實波幅 ATR
            high_low = adj_high - adj_low
            high_close = (adj_high - data[ai_vision_col].shift()).abs()
            low_close = (adj_low - data[ai_vision_col].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.entry_criteria.ATR_LOOKBACK).mean()

            # 3. 動態設定目標與停損價位
            target_profit_price = data[ai_vision_col] + (atr * self.entry_criteria.PROFIT_TARGET_ATR)
            stop_loss_price = data[ai_vision_col] - (atr * self.entry_criteria.STOP_LOSS_ATR)

            hit_target_day = pd.Series(np.inf, index=data.index)
            hit_stop_day = pd.Series(np.inf, index=data.index)

            # 4. 實戰時間迴圈模擬器
            for i in range(1, self.lookahead + 1):
                future_high = adj_high.shift(-i)
                future_low = adj_low.shift(-i)

                target_mask = (future_high >= target_profit_price) & (hit_target_day == np.inf)
                hit_target_day.loc[target_mask] = i

                stop_mask = (future_low <= stop_loss_price) & (hit_stop_day == np.inf)
                hit_stop_day.loc[stop_mask] = i

            # 5. 終極標籤判定：有碰到目標，且比停損早碰到
            target_condition = (hit_target_day != np.inf) & (hit_target_day < hit_stop_day)

            y_all = target_condition.astype(int).values
            y = y_all[self.time_steps - 1:]

            # 確保未來天數足夠才納入訓練
            future_isna = data[ai_vision_col].shift(-self.lookahead).isna().values[self.time_steps - 1:]
            valid_mask = ~future_isna

            X = X[valid_mask]
            y = np.array(y[valid_mask]).astype(int)
            valid_index = aligned_index[valid_mask]
        else:
            y = None
            valid_index = aligned_index

        y_shape_str = str(y.shape) if y is not None else "None"
        dbg.log(f"時序矩陣建立完成！ X 原始形狀: {X.shape}, y 形狀: {y_shape_str}")

        return X, y, valid_index