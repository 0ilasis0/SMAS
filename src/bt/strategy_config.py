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
    min_sentiment_score: int = 4
    block_sell_sentiment_score: int = 8

class PersonaFactory:
    """投資性格工廠：根據使用者選擇，動態產生對應的策略參數"""

    @staticmethod
    def get_config(persona: TradingPersona) -> StrategyConfig:
        if persona == TradingPersona.AGGRESSIVE:
            # 🚀 激進型：關閉防護罩，放寬買進門檻，拉開停利損空間
            return StrategyConfig(
                stop_loss_tolerance=-0.15,        # 容忍 15% 虧損
                trailing_stop_drawdown=-0.10,     # 回落 10% 才跑 (給予更高震盪空間)
                take_profit_target=0.20,          # 賺 20% 才開始停利
                take_profit_sell_ratio=0.30,      # 停利只賣 30%，剩下讓利潤奔跑
                strong_buy_threshold=0.57,        # 稍微看漲就 All-in
                conservative_buy_threshold=0.52,  # 勝率 52% 就敢試水溫
                safe_threshold=0.4,               # 幾乎無視大盤
                cooldown_days=1,
                min_sentiment_score=3,
                max_return_5d=0.4,
                buy_risk=RiskWeights(heavy=0.07, light=0.04)
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                stop_loss_tolerance=-0.05,        # 跌 5% 立刻砍倉
                trailing_stop_drawdown=-0.05,     # 回落 5% 立刻閃人
                take_profit_target=0.05,          # 賺 5% 就跑
                take_profit_sell_ratio=0.75,      # 停利直接賣掉 75%
                strong_buy_threshold=0.63,        # 要求極高勝率才 All-in
                conservative_buy_threshold=0.58,
                safe_threshold=0.50,              # 大盤必須過半安全才出手
                cooldown_days=3,                  # 頻繁休息
                min_sentiment_score=5,
            )

        else:
            # ⚖️ 穩健型：預設值，追求風險與報酬的完美平衡
            return StrategyConfig()