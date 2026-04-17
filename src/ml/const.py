from enum import StrEnum


class TradingMode(StrEnum):
    SWING = "swing"

# ==========================================
# 模型與架構定義
# ==========================================
class ModelCol(StrEnum):
    XGB = "xgb"
    DL = "dl"
    META = "meta"
    MARKET = "market"
    DL_SCALAR = "dl_scalar"

class RNNType(StrEnum):
    """RNN 架構類型"""
    LSTM = "LSTM"
    GRU = "GRU"

class DLModelType(StrEnum):
    """深度學習基礎架構"""
    HYBRID = "HYBRID"      # CNN+RNN
    PURE_CNN = "PURE_CNN"  # 1D-CNN

# ==========================================
# 特徵工程 (Features) 欄位定義
# ==========================================
class FeatureCol(StrEnum):
    """個股模型 (XGBoost/DL) 專用特徵"""
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
    RETURN_5D = "return_5d"

    # K線與量價微結構特徵
    K_UPPER = "k_upper"         # 上影線比例
    K_LOWER = "k_lower"         # 下影線比例
    K_BODY = "k_body"           # 實體K線比例
    BUY_POWER = "buy_power"     # 當日買盤力道 (收盤價位置)
    RS_5D = "rs_5d"             # 個股與大盤的相對強弱

    # 三個不同維度特徵
    OBV = "obv"
    ATR_RATIO = "atr_ratio"
    TREND_STRENGTH = "trend_strength"

    # 標籤 (Label)
    TARGET = "target"

    @classmethod
    def get_features(cls):
        """回傳所有要餵給 AI 學習的特徵欄位清單 (排除 Target)"""
        return [e.value for e in cls if e.value != cls.TARGET.value]

class MarketFeatureCol(StrEnum):
    """大盤防禦模型 (Market Regime) 專用特徵"""
    TWII_BIAS_20 = "twii_bias_20"
    TWII_BIAS_60 = "twii_bias_60"
    TWII_RSI = "twii_rsi"
    TWII_MACD = "twii_macd"
    TWII_VOL_CHG = "twii_vol_chg"

    TWII_ATR_RATIO = "twii_atr_ratio"   # 波動率 (恐慌度)
    SOX_RET_1D = "sox_ret_1d"
    SOX_RET_5D = "sox_ret_5d"
    SOX_TWII_SPREAD = "sox_twii_spread" # 台美相對強弱差
    SOX_CLOSE  = "SOX_close"

    TARGET_DANGER = "target_danger"     # 1:危險(將崩盤), 0:安全

    @classmethod
    def get_features(cls) -> list[str]:
        """自動回傳所有特徵名稱 (排除 Target)"""
        return [e.value for e in cls if e.value != cls.TARGET_DANGER.value]

# ==========================================
# 系統通訊與資料交換定義 (API, Blackboard)
# ==========================================
class SignalCol(StrEnum):
    """ AI 引擎預測訊號 (統一取代舊版的 MetaCol)"""
    PROB_XGB = "prob_xgb"
    PROB_DL = "prob_dl"
    PROB_FINAL = "prob_final"
    PROB_MARKET_SAFE = "prob_market_safe"

class OracleCol(StrEnum):
    """ LLM 情緒分析"""
    SCORE = "sentiment_score"
    REASON = "sentiment_reason"

class QuoteCol(StrEnum):
    """ 市場即時報價與狀態"""
    TICKER = "ticker"
    DATE = "date"
    CURRENT_PRICE = "current_price"
    AVG_5D_VOL = "avg_5d_vol"
    REAL_LATEST_PRICE = "real_latest_price"


class DLParamKey(StrEnum):
    """ 深度學習超參數字典鍵值 (避免字串拼寫錯誤)"""
    BATCH_SIZE = "BATCH_SIZE"
    EPOCHS = "EPOCHS"
    LEARNING_RATE = "LEARNING_RATE"
    CNN_OUT_CHANNELS = "CNN_OUT_CHANNELS"
    LSTM_HIDDEN = "LSTM_HIDDEN"
    DROPOUT = "DROPOUT"

    # 若有其他需要動態覆寫的參數也可以加在這裡
    TIME_STEPS = "TIME_STEPS"
    SCHEDULER_FACTOR = "SCHEDULER_FACTOR"
    SCHEDULER_PATIENCE = "SCHEDULER_PATIENCE"

# ==========================================
# ML 系統全域常數
# ==========================================
class MLConst:
    # 依據：MA_YEAR(240) + DL_TIME_STEPS(20) + 安全緩衝 = 抓取 400 天足矣
    MAX_LOOKBACK = 400

class MLCol(StrEnum):
    N_ESTIMATORS = "n_estimators"