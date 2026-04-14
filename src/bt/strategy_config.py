from dataclasses import dataclass, field
from enum import StrEnum


class TradingPersona(StrEnum):
    AGGRESSIVE = "aggressive"     # 激進型 (追求高報酬，容忍高回撤)
    MODERATE = "moderate"         # 穩健型 (風險報酬平衡)
    CONSERVATIVE = "conservative" # 保守型 (極度厭惡風險，寧可少賺)

@dataclass(frozen=True)
class RiskWeights:
    """定義不同水位下的風險懲罰/敏感係數"""
    heavy: float
    light: float

@dataclass
class StrategyConfig:
    """行為樹策略的基礎超參數容器 (已更新為 Optuna 尋優後的最強 MODERATE 基準)"""

    # ================= 防守與出場參數 =================
    stop_loss_tolerance: float = -0.09        # 強制停損容忍度 (原: -0.1)
    trailing_stop_drawdown: float = -0.10     # 移動停損回落容忍度 (原: -0.06)
    stop_loss_sell_ratio: float = 0.8         # 觸發停損時的賣出比例 (原: 1.0)

    sell_signal_threshold: float = 0.30       # AI 勝率低迷預警門檻 (原: 0.40)
    warning_sell_ratio: float = 0.3           # AI 預警時的戰術減碼比例 (原: 0.5)

    take_profit_target: float = 0.13          # 極端停利目標 (原: 0.10)
    take_profit_sell_ratio: float = 1.0       # 觸發極端停利時的減碼比例 (原: 0.5)

    # ================= 進攻與資金控管參數 =================
    max_entries: int = 2                      # 單一波段最大加碼次數 (原: 3)
    max_gap_ratio: float = 0.04               # 低價股的跳空容忍度 (原: 0.07)

    strong_buy_threshold: float = 0.51        # 強烈買進訊號門檻 (原: 0.6)
    strong_buy_capital_ratio: float = 1.0     # 強烈買進時動用的資金比例 (原: 1.0)

    conservative_buy_threshold: float = 0.50  # 保守買進訊號門檻 (原: 0.55)
    conservative_buy_capital_ratio: float = 0.3 # 保守買進時動用的資金比例 (原: 0.5)

    # ================= 大盤防禦雷達門檻 =================
    safe_threshold: float = 0.36              # 大盤安全度門檻 (原: 0.45)
    cooldown_days: int = 3                    # 交易冷卻天數 (原: 2)

    max_return_5d: float = 0.36               # 5日漲幅過熱門檻 (原: 0.3)
    max_bias_20: float = 0.20                 # 20日乖離率過熱門檻 (原: 0.25)

    # ================= 動態水位風控參數 =================
    # 買進風險懲罰 (原: heavy=0.2, light=0.1)
    buy_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.15, light=0.12))
    # 賣出風險敏感 (原: heavy=0.1, light=0.05)
    sell_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.10, light=0.03))

    # ================= LLM 總開關 =================
    enable_llm_oracle: bool = False
    min_sentiment_score: int = 5
    block_sell_sentiment_score: int = 8

class PersonaFactory:
    """投資性格工廠：根據使用者選擇，動態產生對應的策略參數"""

    @staticmethod
    def get_config(persona: TradingPersona) -> StrategyConfig:
        if persona == TradingPersona.AGGRESSIVE:
            # 🚀 激進型：關閉防護罩，放寬買進門檻，拉開停利損空間
            return StrategyConfig(
                # [防守參數]
                stop_loss_tolerance=-0.20,         # 容忍 20% 虧損 (原: -0.15)
                trailing_stop_drawdown=-0.18,      # 回落 18% 才跑，給予極大震盪空間 (原: -0.10)
                take_profit_target=0.22,           # 賺 22% 才開始停利 (原: 0.20)
                take_profit_sell_ratio=1.0,        # 停利時一次全賣 (原: 0.30)
                stop_loss_sell_ratio=1.0,          # 停損時一次全賣 (原: 未設定，吃預設)
                sell_signal_threshold=0.32,        # AI 預警門檻 (原: 未設定，吃預設)
                warning_sell_ratio=0.70,           # AI 預警時賣出 70% (原: 未設定，吃預設)

                # [進攻參數]
                max_entries=2,                     # 允許加碼 3 次 (原: 預設即為 3)
                max_gap_ratio=0.02,                # 跳空容忍度 (原: 未設定，吃預設)
                strong_buy_threshold=0.50,         # 勝率 49% 就敢重壓 All-in (原: 0.57)
                strong_buy_capital_ratio=0.75,     # 重壓 100% 資金 (原: 未設定，吃預設)
                conservative_buy_threshold=0.48,   # 勝率 48% 就敢試水溫 (原: 0.52)
                conservative_buy_capital_ratio=0.3,# 試水溫只用 30% 資金 (原: 未設定，吃預設)

                # [大盤防禦參數]
                safe_threshold=0.44,               # 幾乎無視大盤，34% 安全度就上 (原: 0.40)
                cooldown_days=3,                   # 停損後冷卻 3 天 (原: 1)
                max_return_5d=0.30,                # 5日漲幅過熱門檻 (原: 0.40)
                max_bias_20=0.20,                  # 20日乖離過熱門檻 (原: 未設定，吃預設)

                # [動態風控水位參數]
                buy_risk=RiskWeights(heavy=0.10, light=0.05), # 倉位重時的買進懲罰 (原: heavy=0.07, light=0.04)
                sell_risk=RiskWeights(heavy=0.11, light=0.10),# 倉位重時的賣出敏感度 (原: 未設定，吃預設)

                # [LLM 參數保留手動設定]
                min_sentiment_score=4,
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                # [防守參數]
                stop_loss_tolerance=-0.08,         # 跌 8% 停損 (原: -0.05，Optuna 認為給予稍微多一點空間較佳)
                trailing_stop_drawdown=-0.07,      # 回落 7% 停損 (原: -0.05)
                take_profit_target=0.09,           # 賺 9% 開始停利 (原: 0.05)
                take_profit_sell_ratio=0.70,       # 停利時賣掉 70% (原: 0.75)
                stop_loss_sell_ratio=1.0,          # 停損時一次全賣 (新增)
                sell_signal_threshold=0.44,        # AI 勝率低於 44% 預警 (新增)
                warning_sell_ratio=0.70,           # 預警時戰術減碼 70% (新增)

                # [進攻參數]
                max_entries=3,                     # 允許加碼 3 次 (新增)
                max_gap_ratio=0.10,                # 較大的跳空容忍度 (新增)
                strong_buy_threshold=0.61,         # 要求 61% 勝率才 All-in (原: 0.63)
                strong_buy_capital_ratio=1.0,      # 強烈買進動用 100% (新增)
                conservative_buy_threshold=0.57,   # 要求 57% 勝率試水溫 (原: 0.58)
                conservative_buy_capital_ratio=0.5,# 保守買進動用 50% (新增)

                # [大盤防禦參數]
                safe_threshold=0.51,               # 大盤安全度大於 51% 才出手 (原: 0.50)
                cooldown_days=1,                   # 冷卻天數縮短為 1 天 (原: 3，增加出手機會)
                max_return_5d=0.38,                # 5日漲幅過熱門檻 (新增)
                max_bias_20=0.29,                  # 20日乖離過熱門檻 (新增)

                # [動態風控水位參數]
                buy_risk=RiskWeights(heavy=0.25, light=0.08),  # 買進風險權重 (新增)
                sell_risk=RiskWeights(heavy=0.09, light=0.01), # 賣出風險權重 (新增)

                # [LLM 參數保留手動設定]
                min_sentiment_score=6,
            )

        else:
            # ⚖️ 穩健型：預設值，追求風險與報酬的完美平衡
            return StrategyConfig()