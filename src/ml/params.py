from dataclasses import dataclass
from enum import StrEnum


@dataclass
class IndicatorParams:
# 均線參數
    MA_WEEK: int = 5
    MA_MONTH: int = 20
    MA_QUARTER: int = 60
    MA_YEAR: int = 240

    # RSI 參數
    RSI_PERIOD: int = 14

    # MACD 參數
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9


class FeatureCol(StrEnum):
    # 均線特徵
    MA_WEEK = "MA_WEEK"
    MA_MONTH = "MA_MONTH"
    MA_QUARTER = "MA_QUARTER"
    MA_YEAR = "MA_YEAR"

    # 技術指標
    RSI = "RSI"
    MACD = "MACD"
    MACD_SIGNAL = "MACD_Signal"

    # 動能特徵
    VOL_CHANGE = "Vol_Change"
    CLOSE_CHANGE = "Close_Change"

    # 標籤 (Label)
    TARGET = "Target"

    @classmethod
    def get_features(cls):
        """回傳所有要餵給 AI 學習的特徵欄位清單 (排除 Target)"""
        return [
            cls.MA_WEEK, cls.MA_MONTH, cls.MA_QUARTER, cls.MA_YEAR,
            cls.RSI, cls.MACD, cls.MACD_SIGNAL,
            cls.VOL_CHANGE, cls.CLOSE_CHANGE
        ]
