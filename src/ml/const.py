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

    # --- 1. 均線特徵 (XGBoost 與 DL 共用的空間感) ---
    BIAS_WEEK = "bias_week"
    BIAS_MONTH = "bias_month"
    BIAS_QUARTER = "bias_quarter"
    BIAS_YEAR = "bias_year"

    # --- 2. 傳統技術指標 (僅 XGBoost 適用) ---
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"

    # --- 3. 轉折與極值特徵 (僅 XGBoost 適用) ---
    KD_K = "kd_k"
    KD_D = "kd_d"
    KD_CROSS = "kd_cross"
    GAP_RATIO = "gap_ratio"

    # --- 4. 大盤相對強弱 (僅 XGBoost 適用) ---
    RS_5D = "rs_5d"
    RS_10D = "rs_10d"

    # --- 5. 量能、波動與 K線幾何 (僅 XGBoost 適用) ---
    # K_UPPER = "k_upper"         # 上影線比例
    # K_LOWER = "k_lower"         # 下影線比例
    # VOL_CHANGE = "vol_change"
    # CLOSE_CHANGE = "close_change"
    OBV = "obv"
    ATR_RATIO = "atr_ratio"

    # --- 6. 原始 DNA (僅 LSTM 適用) ---
    # 這些是 DLFeatureEngine 動態生成的對數報酬率
    OPEN_LOG_CHG = "Open_log_chg"
    HIGH_LOG_CHG = "High_log_chg"
    # LOW_LOG_CHG = "Low_log_chg"
    CLOSE_LOG_CHG = "Adj Close_log_chg"
    VOLUME_LOG_CHG = "Volume_log_chg"
    BB_WIDTH = "bb_width"       # DL 專用波動率縮放

    # --- 7. 標籤 (Label) ---
    TARGET = "target"

    @classmethod
    def get_xgb_features(cls):
        """ XGBoost 專屬：精煉的技術指標與幾何特徵 """
        return [
            cls.BIAS_WEEK.value, cls.BIAS_MONTH.value, cls.BIAS_QUARTER.value, cls.BIAS_YEAR.value,
            cls.MACD.value, cls.MACD_SIGNAL.value,
            cls.KD_K.value, cls.KD_D.value, cls.KD_CROSS.value,
            # cls.VOL_CHANGE.value, cls.CLOSE_CHANGE.value,
            cls.GAP_RATIO.value,
            cls.OBV.value, cls.ATR_RATIO.value,
            # cls.K_UPPER.value, cls.K_LOWER.value,
            cls.RS_5D.value, cls.RS_10D.value
        ]

    @classmethod
    def get_dl_features(cls):
        """ LSTM 專屬：最純粹的正交原始特徵 """
        return [
            cls.OPEN_LOG_CHG.value, cls.HIGH_LOG_CHG.value,
            # cls.LOW_LOG_CHG.value,
            cls.CLOSE_LOG_CHG.value, cls.VOLUME_LOG_CHG.value,
            cls.BIAS_WEEK.value, cls.BIAS_MONTH.value, cls.BB_WIDTH.value
        ]

class MarketFeatureCol(StrEnum):
    """大盤防禦模型 (Market Regime) 專用特徵"""
    TWII_BIAS_20 = "twii_bias_20"
    TWII_BIAS_60 = "twii_bias_60"
    TWII_RSI = "twii_rsi"
    TWII_MACD = "twii_macd"
    TWII_VOL_CHG = "twii_vol_chg"
    TWII_ATR_RATIO = "twii_atr_ratio"   # 波動率 (恐慌度)

    US10Y_SURGE = "us10y_surge"
    FUTURES_OI_LEVEL = "futures_oi_level"

    # 大盤 K 線幾何特徵
    TWII_K_UPPER = "twii_k_upper"
    TWII_K_LOWER = "twii_k_lower"
    TWII_K_BODY = "twii_k_body"

    SOX_RET_1D = "sox_ret_1d"
    SOX_RET_5D = "sox_ret_5d"
    SOX_TWII_SPREAD = "sox_twii_spread" # 台美相對強弱差
    SOX_CLOSE  = "sox_close"

    VIX_CLOSE = "vix_close"
    VIX_SURGE = "vix_surge"
    TWD_DEPRECIATION_5D = "twd_depreciation_5d"

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