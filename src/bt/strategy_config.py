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
    """行為樹策略的基礎超參數容器 (已更新為 PersonaFactory MODERATE 基準)"""

    # ================= 防守與出場參數 =================
    stop_loss_tolerance: float = -0.15         # 容忍 7% 虧損 (原: -0.10)
    trailing_stop_drawdown: float = -0.11      # 移動停損回落 (原: -0.12)
    stop_loss_sell_ratio: float = 0.80         # 停損時賣 80% (原: 1.0)

    sell_signal_threshold: float = 0.34        # AI 預警門檻 (原: 0.38)
    warning_sell_ratio: float = 1.00           # AI 預警時全數下車 (原: 1.0)

    take_profit_target: float = 0.24           # 停利目標 (原: 0.23)
    take_profit_sell_ratio: float = 0.5        # 停利時全數出清 (原: 0.3)

    # ================= [進攻與資金控管參數] =================
    max_entries: int = 1                       # 進場次數 (原: 2)
    max_gap_ratio: float = 0.04                # 跳空容忍度 (原: 0.08)

    strong_buy_threshold: float = 0.52         # 強烈買進門檻 (原: 0.52)
    conservative_buy_threshold: float = 0.48   # 保守買進門檻 (原: 0.49)

    strong_buy_capital_ratio: float = 0.8     # 強勢滿倉重壓 (原: 1.0)
    conservative_buy_capital_ratio: float = 0.5# 保守買進比例 (原: 0.5)

    # ================= [大盤防禦雷達門檻] =================
    safe_threshold: float = 0.55               # 大盤安全度 (原: 0.54)
    cooldown_days: int = 1                     # 交易冷卻天數 (原: 2)

    max_return_5d: float = 0.27                # 5日漲幅上限 (原: 0.16)
    max_bias_20: float = 0.14                  # 20日乖離率上限 (原: 0.11)

    # ================= [動態水位風控參數] =================
    # 買進懲罰：重倉時強迫模型冷靜
    buy_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.15, light=0.10)) # (原: heavy=0.30, light=0.15)
    # 賣出敏感度：調整重倉時的敏感度
    sell_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.12, light=0.03)) # (原: heavy=0.05, light=0.10)

    # ================= [智慧定價參數 (手動設定維持不變)] =================
    buy_panic_discount_atr: float = 0.8
    buy_strong_discount_atr: float = 0.0
    buy_normal_discount_atr: float = 0.4
    sell_strong_premium_atr: float = 0.4
    sell_normal_premium_atr: float = 0.1
    pricing_buy_strong_prob: float = 0.55

    # ================= LLM 總開關 =================
    enable_llm_oracle: bool = False
    min_sentiment_score: int = 5
    block_sell_sentiment_score: int = 8

    '''
    智慧定價與系統防禦參數
    不需要進行調整(optuna)，且大部分參數不同個性應該共用
    '''

    # --- 1. 總經與大盤連動 (Macro & System) ---
    tw_limit_up_ratio: float = 1.099         # 台股漲停板計算比例 (約 10%，保留小數點緩衝)
    tw_limit_down_ratio: float = 0.901       # 台股跌停板計算比例
    sox_surge_threshold: float = 0.015       # 費半漲跌超過 1.5% 啟動開盤位移
    beta_tech: float = 0.4                   # 科技股對費半的連動 Beta 值
    beta_non_tech: float = 0.1               # 非科技股對費半的連動 Beta 值
    market_danger_threshold: float = 0.35    # 大盤安全度低於 35% 視為極度危險

    # --- 2. 智慧買進定價折價幅度 (Buy Pricing Discount ATR) ---
    buy_panic_discount_atr_macro: float = 1.2      # 大盤崩跌時的接刀折價：1.2 倍 ATR
    buy_strong_discount_atr_macro: float = 0.2     # 勝率極高時的追價：折價 0.2 倍 ATR
    buy_normal_discount_atr_macro: float = 0.8     # 常規震盪的低接折價：0.8 倍 ATR
    pricing_buy_extreme_prob: float = 0.75   # 買進：極度看漲門檻

    pricing_buy_strong_prob_macro: float = 0.65    # 買進：強烈看漲門檻 (需搭配情緒)
    pricing_buy_sentiment_min: int = 8       # 買進：強烈看漲所需的情緒底線
    pricing_sell_extreme_prob: float = 0.20  # 賣出：極度看跌/絕望門檻
    pricing_sell_strong_prob: float = 0.70   # 賣出：強勢停利門檻
    buy_rebound_bias: float = -0.06          # 月乖離低於 -6% 視為跌深
    buy_rebound_discount_atr: float = 0.6    # 跌深反彈的承接折價：0.6 倍 ATR

    # --- 3. 智慧賣出定價溢價幅度 (Sell Pricing Premium ATR) ---
    sell_strong_premium_atr_macro: float = 0.6     # 強勢股的優雅出脫：溢價 0.6 倍 ATR
    sell_normal_premium_atr_macro: float = 0.4     # 常規轉弱的反彈調節：溢價 0.4 倍 ATR

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
    wash_risk_win_rate: float = 0.515         # 洗盤警告：勝率低於 51%
    wash_risk_atr_ratio: float = 0.035       # 洗盤警告：日震幅大於 3.5%


class PersonaFactory:
    """投資性格工廠：根據使用者選擇，動態產生對應的策略參數"""

    @staticmethod
    def get_config(persona: TradingPersona) -> StrategyConfig:
        if persona == TradingPersona.AGGRESSIVE:
            # 激進型：策略特徵：單次重擊不加碼、超寬的停利空間讓利潤奔跑、遇到危險分批撤退。
            return StrategyConfig(
                # ================= [防守參數] =================
                stop_loss_tolerance=-0.20,        # 容忍 8% 虧損 (原: -0.19)
                trailing_stop_drawdown=-0.17,     # 高檔回落 17% 就跑 (原: -0.16)
                take_profit_target=0.23,          # 賺 23% 開始分批停利 (原: 0.24)
                take_profit_sell_ratio=0.70,      # 停利時賣 70% (原: 0.3)
                stop_loss_sell_ratio=1.00,        # 停損時賣 100% (原: 0.8)
                sell_signal_threshold=0.32,       # AI 預警門檻 (原: 0.31)
                warning_sell_ratio=0.50,          # AI 預警時減碼 50% (原: 0.7)

                # ================= [進攻參數] =================
                max_entries=2,                    # 限制進場次數 (原: 3)
                max_gap_ratio=0.04,               # 容忍 4% 缺口 (原: 0.10)

                strong_buy_threshold=0.49,        # 強勢買進門檻 (原: 0.49)
                conservative_buy_threshold=0.46,  # 保守買進門檻 (原: 0.46)

                strong_buy_capital_ratio=1.00,    # 強勢重壓 100% (原: 1.0)
                conservative_buy_capital_ratio=0.5,# 保守買進 50% (原: 0.6)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.47,              # 大盤安全度 (原: 0.53)
                cooldown_days=2,                  # 停損後冷卻天數 (原: 1)
                max_return_5d=0.29,               # 5日漲幅上限 (原: 0.32)
                max_bias_20=0.15,                 # 20日乖離上限 (原: 0.28)

                # ================= [動態風控水位參數] =================
                buy_risk=RiskWeights(heavy=0.15, light=0.09), # 買進懲罰 (原: heavy=0.25, light=0.09)
                sell_risk=RiskWeights(heavy=0.06, light=0.04), # 賣出敏感度 (原: heavy=0.14, light=0.09)

                # ================= [智慧定價參數 (手動設定維持不變)] =================
                buy_panic_discount_atr=0.5,
                buy_strong_discount_atr=0.0,
                buy_normal_discount_atr=0.3,
                sell_strong_premium_atr=0.3,
                sell_normal_premium_atr=0.1,
                pricing_buy_strong_prob=0.58,

                # ================= [LLM 參數保留手動設定] =================
                min_sentiment_score=4,
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                # ================= [防守參數] =================
                stop_loss_tolerance=-0.08,        # 容忍 8% 虧損 (原: -0.15)
                trailing_stop_drawdown=-0.08,     # 行動停損回落 8% (原: -0.08)
                take_profit_target=0.12,          # 賺 12% 開始分批停利 (原: 0.15)
                take_profit_sell_ratio=0.70,      # 停利時賣出 70% (原: 0.30)
                stop_loss_sell_ratio=0.80,        # 停損時賣出 80% (原: 1.0)
                sell_signal_threshold=0.32,       # AI 預警門檻 (原: 0.44)
                warning_sell_ratio=1.00,          # AI 預警時 100% 清倉逃命 (原: 1.0)

                # ================= [進攻參數] =================
                max_entries=1,                    # 限制進場次數 (原: 4)
                max_gap_ratio=0.09,               # 跳空容忍度 (原: 0.09)

                strong_buy_threshold=0.56,        # 強勢買進門檻 (原: 0.54)
                conservative_buy_threshold=0.52,  # 保守買進門檻 (原: 0.50)

                strong_buy_capital_ratio=1.0,    # 強勢買進資金比例 (原: 1.0)
                conservative_buy_capital_ratio=0.60, # 保守買進資金比例 (原: 0.4)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.55,              # 大盤安全度 (原: 0.54)
                cooldown_days=1,                  # 停損後冷卻天數 (原: 5)
                max_return_5d=0.26,               # 5日漲幅上限 (原: 0.23)
                max_bias_20=0.14,                 # 20日乖離率上限 (原: 0.27)

                # ================= [動態風控水位參數] =================
                # 買進風險權重 (原: heavy=0.15, light=0.06)
                buy_risk=RiskWeights(heavy=0.15, light=0.08),
                # 賣出風險權重 (原: heavy=0.06, light=0.05)
                sell_risk=RiskWeights(heavy=0.16, light=0.09),

                # ================= [定價參數 (維持您原本的手動設定)] =================
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