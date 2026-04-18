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

        # 🌟 新增：高低點轉折與極端情緒特徵 (專攻摸底與突破)
        # 1. 區間相對位置 (Donchian Channels): 判斷破底或創高
        donchian_w = self.params.DONCHIAN_WINDOW
        high_nd = data[StockCol.HIGH].rolling(window=donchian_w).max()
        low_nd = data[StockCol.LOW].rolling(window=donchian_w).min()

        # 距離區間高低點的百分比 (越接近 0 代表越接近極值)
        data[FeatureCol.DIST_TO_20D_HIGH] = (high_nd - data[StockCol.HIGH]) / (high_nd + 1e-9)
        data[FeatureCol.DIST_TO_20D_LOW] = (data[StockCol.LOW] - low_nd) / (low_nd + 1e-9)

        # 2. KD 隨機指標: 判斷極度超買超賣的反轉
        kd_rsv_w = self.params.KD_RSV_WINDOW
        kd_com = self.params.KD_SMOOTH - 1  # 轉換為 ewm 需要的 com 參數

        # RSV = (今日收盤 - 近N日最低) / (近N日最高 - 近N日最低)
        low_kd = data[StockCol.LOW].rolling(window=kd_rsv_w).min()
        high_kd = data[StockCol.HIGH].rolling(window=kd_rsv_w).max()
        rsv = (data[StockCol.CLOSE] - low_kd) / (high_kd - low_kd + 1e-9) * 100

        # KD 需要遞迴計算，我們用 ewm 模擬平滑
        data[FeatureCol.KD_K] = rsv.ewm(com=kd_com, adjust=False).mean()
        data[FeatureCol.KD_D] = data[FeatureCol.KD_K].ewm(com=kd_com, adjust=False).mean()

        # KD 開口差值 (負轉正代表黃金交叉，強烈買訊)
        data[FeatureCol.KD_CROSS] = data[FeatureCol.KD_K] - data[FeatureCol.KD_D]

        # 3. 恐慌/突破缺口 (Gap Ratio)
        # 今日開盤跳空幅度 (若大於 0.02 代表強勢跳空，小於 -0.02 代表恐慌跳空)
        prev_close = data[StockCol.CLOSE].shift(1)
        data[FeatureCol.GAP_RATIO] = (data[StockCol.OPEN] - prev_close) / (prev_close + 1e-9)

        data[FeatureCol.VOL_CHANGE] = data[StockCol.VOLUME].pct_change()
        data[FeatureCol.CLOSE_CHANGE] = data[ai_vision_col].pct_change()
        data[FeatureCol.RETURN_5D] = data[ai_vision_col].pct_change(periods=5)
        data[FeatureCol.RETURN_10D] = data[ai_vision_col].pct_change(periods=10)

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
            # 原本的 5D 相對強弱
            stock_ma5 = data[ai_vision_col].rolling(window=5).mean()
            stock_ma20 = ma_m
            stock_momentum = (stock_ma5 - stock_ma20) / (stock_ma20 + 1e-9)

            twii_ma5 = data[twii_close_col].rolling(window=5).mean()
            twii_ma20 = data[twii_close_col].rolling(window=20).mean()
            twii_momentum = (twii_ma5 - twii_ma20) / (twii_ma20 + 1e-9)
            data[FeatureCol.RS_5D] = stock_momentum - twii_momentum

            # 10D 相對強弱
            stock_ma10 = data[ai_vision_col].rolling(window=10).mean()
            stock_momentum_10 = (stock_ma10 - stock_ma20) / (stock_ma20 + 1e-9)

            twii_ma10 = data[twii_close_col].rolling(window=10).mean()
            twii_momentum_10 = (twii_ma10 - twii_ma20) / (twii_ma20 + 1e-9)
            data[FeatureCol.RS_10D] = stock_momentum_10 - twii_momentum_10
        else:
            data[FeatureCol.RS_5D] = 0.0
            data[FeatureCol.RS_10D] = 0.0

        return data

    def _create_labels(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        if df.empty: return df
        data = df.copy()
        ai_vision_col = str(StockCol.ADJ_CLOSE)

        adj_factor = data[ai_vision_col] / (data[StockCol.CLOSE] + 1e-9)
        adj_high = data[StockCol.HIGH] * adj_factor
        adj_low = data[StockCol.LOW] * adj_factor

        # 1. 計算真實波幅 ATR
        high_low = adj_high - adj_low
        high_close = (adj_high - data[ai_vision_col].shift()).abs()
        low_close = (adj_low - data[ai_vision_col].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.entry_criteria.ATR_LOOKBACK).mean()

        # 2. 動態設定目標與停損價位
        target_atr = self.entry_criteria.PROFIT_TARGET_ATR
        stop_atr = self.entry_criteria.STOP_LOSS_ATR

        target_profit_price = data[ai_vision_col] + (atr * target_atr)
        stop_loss_price = data[ai_vision_col] - (atr * stop_atr)

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

        # 4. 終極連續獎勵引擎
        # 取得最後一天 (第 lookahead 天) 的收盤價來結算未達標的獲利
        final_day_close = data[ai_vision_col].shift(-lookahead)

        # 結算時的實際 ATR 獲利倍數
        realized_atr_multiple = (final_day_close - data[ai_vision_col]) / (atr + 1e-9)

        # 初始化獎勵陣列
        reward_scores = pd.Series(np.nan, index=data.index)

        # 條件 A：先碰到停損 (最差情況) -> 給予 0.0 分
        mask_stop_first = (hit_stop_day != np.inf) & (hit_stop_day <= hit_target_day)
        reward_scores.loc[mask_stop_first] = 0.0

        # 條件 B：先碰到完美目標 (提早達標) -> 給予滿分 1.0 分
        mask_target_first = (hit_target_day != np.inf) & (hit_target_day < hit_stop_day)
        reward_scores.loc[mask_target_first] = 1.0

        # 條件 C：時間到期 (10天都沒碰到停利與停損) -> 依據最終獲利給予線性獎勵
        mask_timeout = (hit_target_day == np.inf) & (hit_stop_day == np.inf)

        # 處理到期結算的線性分數 (使用 np.clip 限制在 0~1 之間)
        # 公式：將實際獲利倍數，映射到分數。
        # 舉例：如果 realized_atr_multiple 是 1.5 (PROFIT的一半)，分數就是 1.5 / 3.0 = 0.5 分。
        timeout_scores = (realized_atr_multiple.loc[mask_timeout] / target_atr).clip(lower=0.0, upper=1.0)

        # 稍微對未獲利(跌)的進行一點容忍度懲罰，如果虧損，就給 0.1~0.2，避免完全跟停損一樣是 0
        timeout_scores[realized_atr_multiple.loc[mask_timeout] <= 0] = 0.1

        reward_scores.loc[mask_timeout] = timeout_scores

        # 過濾未來資料尚不足夠的尾端天數
        valid_future_mask = data[ai_vision_col].shift(-lookahead).notna()

        # 🌟 這裡不再轉換成 Int64，因為現在是 0.0 ~ 1.0 的浮點數了！
        data[FeatureCol.TARGET] = reward_scores.astype(float)
        data.loc[~valid_future_mask, FeatureCol.TARGET] = pd.NA

        return data