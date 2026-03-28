from enum import StrEnum


class RNNType(StrEnum):
    LSTM = "LSTM"
    GRU = "GRU"


class FeatureCol(StrEnum):
    # 均線特徵
    BIAS_WEEK = "bias_week"
    BIAS_MONTH = "bias_month"
    BIAS_QUARTER = "bias_quarter"
    BIAS_YEAR = "bias_year"

    # 技術指標
    RSI = "rsi"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"

    # 動能特徵
    VOL_CHANGE = "vol_change"
    CLOSE_CHANGE = "close_change"
    BB_WIDTH = "bb_width"

    # 標籤 (Label)
    TARGET = "target"

    @classmethod
    def get_features(cls):
        """回傳所有要餵給 AI 學習的特徵欄位清單 (排除 Target)"""
        return [
            cls.BIAS_WEEK, cls.BIAS_MONTH, cls.BIAS_QUARTER, cls.BIAS_YEAR,
            cls.RSI, cls.MACD, cls.MACD_SIGNAL,
            cls.VOL_CHANGE, cls.CLOSE_CHANGE, cls.BB_WIDTH
        ]

class MetaCol(StrEnum):
    """Meta-Learner (Level 1) 專用的欄位名稱"""
    PROB_XGB = "prob_xgb"
    PROB_DL = "prob_dl"
    PROB_FINAL = "prob_final"
    # 直接引用 FeatureCol 的 TARGET，確保一致性
    TARGET = FeatureCol.TARGET

class MLConst:
    # 依據：MA_YEAR(240) + DL_TIME_STEPS(20) + 安全緩衝 = 抓取 300~500 天足矣
    MAX_LOOKBACK = 400
