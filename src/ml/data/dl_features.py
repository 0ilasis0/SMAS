import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data.const import StockCol
from debug import dbg
from ml.const import FeatureCol
from ml.params import DLHyperParams, EntryQualityCriteria, IndicatorParams


class DLFeatureEngine:
    """
    專為時序深度學習模型 (CNN/LSTM) 設計的特徵工程 (原始 DNA 視角 - Path B)。
    僅負責生成原始正交特徵與 3D 滑動視窗，正規化 (Scaling) 交由 Trainer 在 CV 迴圈內處理以防止未來資料外洩。
    """
    def __init__(
            self,
            lookahead: int,
            time_steps: int = DLHyperParams.time_steps,
            entry_criteria: EntryQualityCriteria = EntryQualityCriteria()
        ):
        self.lookahead = lookahead
        self.time_steps = time_steps
        self.entry_criteria = entry_criteria
        # 額外的 19 天歷史緩衝(暖機期)
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

        active_features = FeatureCol.get_dl_features()

        # 拿「過去」補「現在」，確保時序連續性
        data = df.copy().ffill()

        # 1. 還原並計算 OHLCV 的純粹對數報酬率 (模型的核心時序 DNA)
        adj_factor = data[StockCol.ADJ_CLOSE] / (data[StockCol.CLOSE] + 1e-9)

        data[FeatureCol.OPEN_LOG_CHG] = np.log((data[StockCol.OPEN] * adj_factor) / (data[StockCol.OPEN].shift(1) * adj_factor.shift(1) + 1e-9))
        data[FeatureCol.HIGH_LOG_CHG] = np.log((data[StockCol.HIGH] * adj_factor) / (data[StockCol.HIGH].shift(1) * adj_factor.shift(1) + 1e-9))
        data[FeatureCol.CLOSE_LOG_CHG] = np.log((data[StockCol.ADJ_CLOSE]) / (data[StockCol.ADJ_CLOSE].shift(1) + 1e-9))
        data[FeatureCol.VOLUME_LOG_CHG] = np.log1p(data[StockCol.VOLUME]) - np.log1p(data[StockCol.VOLUME].shift(1))

        # 2. 輔助大局特徵：趨勢乖離與通道寬度 (提供絕對空間位置與波動縮放感)
        ma_w = data[StockCol.ADJ_CLOSE].rolling(window=IndicatorParams.MA_WEEK).mean()
        ma_m = data[StockCol.ADJ_CLOSE].rolling(window=IndicatorParams.MA_MONTH).mean()
        rolling_std = data[StockCol.ADJ_CLOSE].rolling(window=IndicatorParams.MA_MONTH).std()

        data[FeatureCol.BIAS_WEEK] = (data[StockCol.ADJ_CLOSE] - ma_w) / (ma_w + 1e-9)
        data[FeatureCol.BIAS_MONTH] = (data[StockCol.ADJ_CLOSE] - ma_m) / (ma_m + 1e-9)
        data[FeatureCol.BB_WIDTH] = (rolling_std * 2) / (ma_m + 1e-9)

        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=active_features)

        if len(data) < self.time_steps:
            dbg.war("扣除暖機期與無效資料後，資料量不足以建立滑動視窗。")
            return None, None, None

        # 3. 建立 3D 滑動視窗矩陣 (Batch, Time_Steps, Features)
        raw_features = data[active_features].values
        X = sliding_window_view(raw_features, window_shape=self.time_steps, axis=0)
        X = np.transpose(X, (0, 2, 1))

        aligned_index = data.index[self.time_steps - 1:]

        # 4. 標籤生成區塊
        if is_training:
            # 還原高低點用於精準的屏障觸發判定
            current_adj_factor = data[StockCol.ADJ_CLOSE] / (data[StockCol.CLOSE] + 1e-9)
            adj_high = data[StockCol.HIGH] * current_adj_factor
            adj_low = data[StockCol.LOW] * current_adj_factor

            # 計算真實波幅 ATR
            high_low = adj_high - adj_low
            high_close = (adj_high - data[StockCol.ADJ_CLOSE].shift()).abs()
            low_close = (adj_low - data[StockCol.ADJ_CLOSE].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.entry_criteria.ATR_LOOKBACK).mean()

            # 動態設定停利目標 (1.6x ATR) 與停損價位 (1.5x ATR)
            target_profit_price = data[StockCol.ADJ_CLOSE] + (atr * self.entry_criteria.PROFIT_TARGET_ATR)
            stop_loss_price = data[StockCol.ADJ_CLOSE] - (atr * self.entry_criteria.STOP_LOSS_ATR)

            hit_target_day = pd.Series(np.inf, index=data.index)
            hit_stop_day = pd.Series(np.inf, index=data.index)

            # 實戰時間迴圈模擬器
            for i in range(1, self.lookahead + 1):
                future_high = adj_high.shift(-i)
                future_low = adj_low.shift(-i)

                target_mask = (future_high >= target_profit_price) & (hit_target_day == np.inf)
                hit_target_day.loc[target_mask] = i

                stop_mask = (future_low <= stop_loss_price) & (hit_stop_day == np.inf)
                hit_stop_day.loc[stop_mask] = i

            # ➔ 貫徹特種部隊標籤法：只有先碰到目標、且未碰到停損才算成功突破
            target_condition = (hit_target_day != np.inf) & (hit_target_day < hit_stop_day)

            y_all = target_condition.astype(int).values
            y = y_all[self.time_steps - 1:]

            # 濾除未來時間不足 lookahead 天的尾端資料
            future_isna = data[StockCol.ADJ_CLOSE].shift(-self.lookahead).isna().values[self.time_steps - 1:]
            valid_mask = ~future_isna

            X = X[valid_mask]
            y = np.array(y[valid_mask]).astype(int)
            valid_index = aligned_index[valid_mask]
        else:
            y = None
            valid_index = aligned_index

        y_shape_str = str(y.shape) if y is not None else "None"
        dbg.log(f"時序矩陣建立完成！ X 原始形狀: {X.shape}, y 最終標籤形狀: {y_shape_str}")

        return X, y, valid_index