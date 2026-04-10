from dataclasses import dataclass
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
    """行為樹策略的基礎超參數容器"""
    stop_loss_tolerance: float = -0.05       # 強制停損容忍度 (-5%)
    trailing_stop_drawdown: float = -0.08    # 移動停損回落容忍度 (-8%)
    stop_loss_sell_ratio: float = 1.0        # 觸發停損時的賣出比例 (100% 全面撤退)

    sell_signal_threshold: float = 0.40      # AI 勝率低迷預警門檻 (<40%)
    warning_sell_ratio: float = 0.5          # AI 預警時的戰術減碼比例 (賣出 50%)

    take_profit_target: float = 0.30         # 極端停利目標 (+30%)
    take_profit_sell_ratio: float = 0.5      # 觸發極端停利時的減碼比例 (先入袋 50%)

    # ================= 進攻與資金控管參數 =================
    max_entries: int = 3                     # 單一波段最大加碼次數
    max_gap_ratio: float = 0.07              # 低價股的跳空容忍度

    strong_buy_threshold: float = 0.75       # 強烈買進訊號門檻 (>= 75%)
    strong_buy_capital_ratio: float = 1.0    # 強烈買進時動用的資金比例 (100%)

    conservative_buy_threshold: float = 0.60 # 保守買進訊號門檻 (>= 60%)
    conservative_buy_capital_ratio: float = 0.5 # 保守買進時動用的資金比例 (50%)

    # ================= 大盤防禦雷達門檻 =================
    safe_threshold: float = 0.45
    cooldown_days: int = 2

    max_return_5d: float = 0.3
    max_bias_20: float = 0.25

    # ================= 動態水位風控參數 (New) =================
    # 數值越高，持股比例對門檻的「懲罰」就越重
    buy_risk: RiskWeights = RiskWeights(heavy=0.2, light=0.1)       # 買進時的倉位懲罰係數
    sell_risk: RiskWeights = RiskWeights(heavy=0.1, light=0.05)     # 賣出時的倉位敏感係數 (放寬門檻)

    # LLM 總開關
    enable_llm_oracle: bool = False
    min_sentiment_score: int = 4
    block_sell_sentiment_score: int = 8


class PersonaFactory:
    """投資性格工廠：根據使用者選擇，動態產生對應的策略參數"""

    @staticmethod
    def get_config(persona: TradingPersona) -> StrategyConfig:
        if persona == TradingPersona.AGGRESSIVE:
            # 激進型：關閉防護罩，放寬買進門檻，拉開停利損空間
            return StrategyConfig(
                stop_loss_tolerance=-0.08,        # 容忍 8% 虧損
                trailing_stop_drawdown=-0.08,     # 回落 8% 才跑
                take_profit_target=0.10,          # 賺 10% 才開始停利
                strong_buy_threshold=0.55,        # 稍微看漲就 All-in
                conservative_buy_threshold=0.53,  # 勝率 52% 就敢試水溫
                safe_threshold=0.35,              # 幾乎無視大盤
                cooldown_days=1,
                min_sentiment_score=3,
                max_return_5d = 0.4,
                buy_risk = RiskWeights(heavy=0.07, light=0.04)
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                stop_loss_tolerance=-0.025,       # 跌 2.5% 立刻砍倉
                trailing_stop_drawdown=-0.025,    # 回落 2.5% 立刻閃人
                take_profit_target=0.03,          # 賺 3% 就跑
                take_profit_sell_ratio=1.0,       # 停利直接全賣，不留戀
                strong_buy_threshold=0.6,        # 要求極高勝率才 All-in
                conservative_buy_threshold=0.63,
                safe_threshold=0.65,              # 大盤安全度必須大於 65% 才准買！
                cooldown_days=4,
                min_sentiment_score=5,
            )

        else:
            return StrategyConfig()
