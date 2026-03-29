from dataclasses import dataclass
from enum import StrEnum


class TradingPersona(StrEnum):
    AGGRESSIVE = "aggressive"     # 激進型 (追求高報酬，容忍高回撤)
    MODERATE = "moderate"         # 穩健型 (風險報酬平衡)
    CONSERVATIVE = "conservative" # 保守型 (極度厭惡風險，寧可少賺)

@dataclass
class StrategyConfig:
    """行為樹策略的基礎超參數容器"""
    # --- 防守與風險控管 ---
    stop_loss_tolerance: float = -0.04
    trailing_stop_drawdown: float = -0.04
    stop_loss_sell_ratio: float = 1.0

    sell_signal_threshold: float = 0.52
    warning_sell_ratio: float = 1.0

    take_profit_target: float = 0.05
    take_profit_sell_ratio: float = 0.5

    # --- 進攻與資金控管 ---
    max_entries: int = 2
    max_gap_ratio: float = 0.03

    strong_buy_threshold: float = 0.60
    strong_buy_capital_ratio: float = 1.0

    conservative_buy_threshold: float = 0.57
    conservative_buy_capital_ratio: float = 0.5

    # 大盤防禦雷達門檻
    safe_threshold: float = 0.50
    cooldown_days: int = 5

class PersonaFactory:
    """投資性格工廠：根據使用者選擇，動態產生對應的策略參數"""

    @staticmethod
    def get_config(persona: TradingPersona) -> StrategyConfig:
        if persona == TradingPersona.AGGRESSIVE:
            # 🔥 激進型：關閉防護罩，放寬買進門檻，拉開停利損空間
            return StrategyConfig(
                stop_loss_tolerance=-0.08,        # 容忍 8% 虧損
                trailing_stop_drawdown=-0.08,     # 回落 8% 才跑
                take_profit_target=0.10,          # 賺 10% 才開始停利
                strong_buy_threshold=0.55,        # 稍微看漲就 All-in
                conservative_buy_threshold=0.52,  # 勝率 52% 就敢試水溫
                safe_threshold=0.20,              # 幾乎無視大盤 (除非大盤極度恐慌低於20%)
                cooldown_days=1                   # 停損後隔天立刻又想進場
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                stop_loss_tolerance=-0.025,       # 跌 2.5% 立刻砍倉
                trailing_stop_drawdown=-0.025,    # 回落 2.5% 立刻閃人
                take_profit_target=0.03,          # 賺 3% 就跑
                take_profit_sell_ratio=1.0,       # 停利直接全賣，不留戀
                strong_buy_threshold=0.65,        # 要求極高勝率才 All-in
                conservative_buy_threshold=0.60,
                safe_threshold=0.65,              # 🚀 大盤安全度必須大於 65% 才准買！
                cooldown_days=7                   # 停損後冷靜一整個禮拜
            )

        else:
            return StrategyConfig()
