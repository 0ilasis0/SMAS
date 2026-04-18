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
    stop_loss_tolerance: float = -0.10           # 強制停損容忍度 (原: -0.09)
    trailing_stop_drawdown: float = -0.1         # 移動停損回落容忍度 (原: -0.10)
    stop_loss_sell_ratio: float = 0.8            # 觸發停損時的賣出比例 (原: 0.8)

    sell_signal_threshold: float = 0.27          # AI 勝率低迷預警門檻 (原: 0.30)
    warning_sell_ratio: float = 0.7              # AI 預警時的戰術減碼比例 (原: 0.3)

    take_profit_target: float = 0.18             # 極端停利目標 (原: 0.13)
    take_profit_sell_ratio: float = 1.0          # 觸發極端停利時的減碼比例 (原: 1.0)

    # ================= 進攻與資金控管參數 =================
    max_entries: int = 1                         # 單一波段最大加碼次數 (原: 2)
    max_gap_ratio: float = 0.05                  # 低價股的跳空容忍度 (原: 0.04)

    strong_buy_threshold: float = 0.55           # 強烈買進訊號門檻 (原: 0.51)
    strong_buy_capital_ratio: float = 0.7        # 強烈買進時動用的資金比例 (原: 1.0)

    conservative_buy_threshold: float = 0.5      # 保守買進訊號門檻 (原: 0.50)
    conservative_buy_capital_ratio: float = 0.40 # 保守買進時動用的資金比例 (原: 0.3)

    # ================= 大盤防禦雷達門檻 =================
    safe_threshold: float = 0.45                 # 大盤安全度門檻 (原: 0.44)
    cooldown_days: int = 3                       # 交易冷卻天數

    max_return_5d: float = 0.23                  # 5日漲幅過熱門檻 (原: 0.36)
    max_bias_20: float = 0.2                     # 20日乖離率過熱門檻 (原: 0.20)

    # ================= 動態水位風控參數 =================
    # 買進風險懲罰 (原: heavy=0.15, light=0.12)
    buy_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.15, light=0.1))
    # 賣出風險敏感 (原: heavy=0.10, light=0.03)
    sell_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.1, light=0.05))

    # ================= LLM 總開關 =================
    enable_llm_oracle: bool = False
    min_sentiment_score: int = 5
    block_sell_sentiment_score: int = 8

    '''
    智慧定價與系統防禦參數 (Smart Pricing & Risk)
    大部分不會調整，且不同個性應該共用(有用空格分開前的為可調整)
    '''

    # --- 1. 總經與大盤連動 (Macro & System) ---
    tw_limit_up_ratio: float = 1.099         # 台股漲停板計算比例 (約 10%，保留小數點緩衝)
    tw_limit_down_ratio: float = 0.901       # 台股跌停板計算比例
    sox_surge_threshold: float = 0.015       # 費半漲跌超過 1.5% 啟動開盤位移
    beta_tech: float = 0.4                   # 科技股對費半的連動 Beta 值
    beta_non_tech: float = 0.1               # 非科技股對費半的連動 Beta 值
    market_danger_threshold: float = 0.35    # 大盤安全度低於 35% 視為極度危險

    # --- 2. 智慧買進定價折價幅度 (Buy Pricing Discount ATR) ---
    buy_panic_discount_atr: float = 1.2      # 大盤崩跌時的接刀折價：1.2 倍 ATR
    buy_strong_discount_atr: float = 0.2     # 勝率極高時的追價：折價 0.2 倍 ATR
    buy_normal_discount_atr: float = 0.8     # 常規震盪的低接折價：0.8 倍 ATR
    pricing_buy_extreme_prob: float = 0.75   # 買進：極度看漲門檻

    pricing_buy_strong_prob: float = 0.65    # 買進：強烈看漲門檻 (需搭配情緒)
    pricing_buy_sentiment_min: int = 8       # 買進：強烈看漲所需的情緒底線
    pricing_sell_extreme_prob: float = 0.20  # 賣出：極度看跌/絕望門檻
    pricing_sell_strong_prob: float = 0.70   # 賣出：強勢停利門檻
    buy_rebound_bias: float = -0.06          # 月乖離低於 -6% 視為跌深
    buy_rebound_discount_atr: float = 0.6    # 跌深反彈的承接折價：0.6 倍 ATR

    # --- 3. 智慧賣出定價溢價幅度 (Sell Pricing Premium ATR) ---
    sell_strong_premium_atr: float = 0.6     # 強勢股的優雅出脫：溢價 0.6 倍 ATR
    sell_normal_premium_atr: float = 0.4     # 常規轉弱的反彈調節：溢價 0.4 倍 ATR

    sell_panic_discount_atr: float = 0.5     # 停損或看壞時的逃命折價：0.5 倍 ATR (注意這是折價賤賣)
    sell_overheated_bias: float = 0.08       # 月乖離大於 8% 視為超漲
    sell_overheated_premium_atr: float = 0.8 # 超漲時的掛高出貨：溢價 0.8 倍 ATR
    earnings_shield_days: int = 3            # 距離法說會幾天內禁止買進

    # --- 4. 觀望與洗盤警告門檻 (Hold & Warning Thresholds) ---
    hold_danger_threshold: float = 0.4       # 大盤低於 40% 建議持有部位避險
    hold_weak_threshold: float = 0.3         # 勝率低於 30% 且空手時建議觀望
    hold_neutral_threshold: float = 0.6      # 勝率低於 60% 且空手時建議保留現金
    hold_wait_threshold: float = 0.4         # 勝率大於 40% 且有部位時建議續抱

    # --- 5. 大盤警告門檻 ---
    wash_risk_win_rate: float = 0.51         # 洗盤警告：勝率低於 51%
    wash_risk_atr_ratio: float = 0.035       # 洗盤警告：日震幅大於 3.5%


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
                take_profit_target=0.25,           # 賺 25% 才開始停利 (原: 0.20)
                take_profit_sell_ratio=1.0,        # 停利時一次全賣 (原: 0.30)
                stop_loss_sell_ratio=1.0,          # 停損時一次全賣
                sell_signal_threshold=0.32,        # AI 預警門檻
                warning_sell_ratio=0.70,           # AI 預警時賣出 70%

                # [進攻參數]
                max_entries=3,                     # 允許加碼 3 次
                max_gap_ratio=0.02,                # 跳空容忍度
                strong_buy_threshold=0.50,         # 勝率 50% 就敢重壓 All-in (原: 0.57)
                strong_buy_capital_ratio=0.75,     # 重壓 100 % 資金
                conservative_buy_threshold=0.48,   # 勝率 48% 就敢試水溫 (原: 0.52)
                conservative_buy_capital_ratio=0.3,# 試水溫只用 30% 資金

                # [大盤防禦參數]
                safe_threshold=0.44,               # 幾乎無視大盤，44% 安全度就上 (原: 0.40)
                cooldown_days=3,                   # 停損後冷卻 3 天 (原: 1)
                max_return_5d=0.30,                # 5日漲幅過熱門檻 (原: 0.40)
                max_bias_20=0.20,                  # 20日乖離過熱門檻

                # [動態風控水位參數]
                buy_risk=RiskWeights(heavy=0.10, light=0.05), # 倉位重時的買進懲罰 (原: heavy=0.07, light=0.04)
                sell_risk=RiskWeights(heavy=0.11, light=0.10),# 倉位重時的賣出敏感度

                # [定價參數]
                buy_panic_discount_atr=0.5,      # 別人恐慌接刀要折價 1.2，激進型只要折價 0.5 就敢搶反彈
                buy_strong_discount_atr=0.0,     # 勝率高時，直接掛「平盤價 (0 折價)」全力追擊
                buy_normal_discount_atr=0.3,     # 常規震盪也只等微幅拉回 0.3 ATR 就買
                sell_strong_premium_atr=0.3,     # 賣出時不想等太久，溢價 0.3 ATR 就願意賣給別人
                sell_normal_premium_atr=0.1,     # 常規轉弱時，幾乎平盤就趕快逃命
                pricing_buy_strong_prob=0.58,    # 只要勝率 58% 就認定是強勢股，啟動追價模式

                # [LLM 參數保留手動設定]
                min_sentiment_score=4,
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                # [防守參數]
                stop_loss_tolerance=-0.08,         # 跌 8% 停損
                trailing_stop_drawdown=-0.08,      # 回落 8% 停損 (原: -0.05)
                take_profit_target=0.15,           # 賺 11% 開始停利 (原: 0.10)
                take_profit_sell_ratio=0.75,       # 停利時賣掉 50% (原: 0.50)
                stop_loss_sell_ratio=0.75,         # 停損時賣75% (原: 1.00)
                sell_signal_threshold=0.44,        # AI 勝率低於 25% 預警 (原: 0.44)
                warning_sell_ratio=0.75,           # 預警時戰術減碼 50% (原: 1.00，不再一次性清倉)

                # [進攻參數]
                max_entries=1,                     # 允許加碼 1 次 (原: 2)
                max_gap_ratio=0.03,                # 跳空容忍度極低，不買跳空股 (原: 0.07)
                strong_buy_threshold=0.59,         # 要求 61% 勝率才重倉 (原: 0.59)
                strong_buy_capital_ratio=1.0,     # 強烈買進動用 80% 資金 (原: 0.80)
                conservative_buy_threshold=0.54,   # 要求 54% 勝率試水溫 (原: 0.52)
                conservative_buy_capital_ratio=0.40,# 保守買進動用 40% 資金

                # [大盤防禦參數]
                safe_threshold=0.47,               # 大盤安全度大於 47% 才出手
                cooldown_days=3,                   # 冷卻天數維持 3 天
                max_return_5d=0.16,                # 5日漲幅過熱門檻，大降溫不追高 (原: 0.37)
                max_bias_20=0.23,                  # 20日乖離過熱門檻 (原: 0.15)

                # [動態風控水位參數]
                buy_risk=RiskWeights(heavy=0.10, light=0.14),  # 買進風險權重 (原: heavy=0.20, light=0.09)
                sell_risk=RiskWeights(heavy=0.11, light=0.09), # 賣出風險權重 (原: heavy=0.05, light=0.04)

                # [定價參數]
                buy_panic_discount_atr=1.8,      # 大盤恐慌時，掛在極度深淵 (折價 1.8 ATR) 等天上掉禮物
                buy_strong_discount_atr=0.5,     # 就算勝率極高，也堅持要拉回 0.5 ATR 才肯買
                buy_normal_discount_atr=1.0,     # 常規震盪時，掛在地板價 (折價 1.0 ATR) 死等
                sell_strong_premium_atr=1.0,     # 賣出時獅子大開口，掛高高 (溢價 1.0 ATR) 慢慢等有緣人
                sell_normal_premium_atr=0.8,     # 常規轉弱也堅持要賣個好價錢
                pricing_buy_extreme_prob=0.85,   # 要求勝率高達 85% 才肯承認是極度看漲

                # [LLM 參數保留手動設定]
                min_sentiment_score=6,
            )

        else:
            # ⚖️ 穩健型：預設值，追求風險與報酬的完美平衡
            return StrategyConfig()