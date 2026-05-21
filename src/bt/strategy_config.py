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
    stop_loss_tolerance: float = -0.10         # 強制停損容忍度，回到穩健的 10% (原: -0.09)
    trailing_stop_drawdown: float = -0.12      # 移動停損回落容忍度維持不變 (原: -0.12)
    stop_loss_sell_ratio: float = 1.0          # Optuna 建議：停損時直接 100% 清倉，不留殘存部位 (原: 0.80)

    sell_signal_threshold: float = 0.38        # AI 預警門檻大幅提高，稍微轉弱就準備跑 (原: 0.28)
    warning_sell_ratio: float = 1.0            # AI 一旦發出預警，直接 100% 下車避險 (原: 0.3)

    take_profit_target: float = 0.23           # 停利目標進一步拉高至 23% (原: 0.20)
    take_profit_sell_ratio: float = 0.3        # 停利時依然只賣 30%，保留核心倉位賺波段 (原: 0.3)

    # ================= [進攻與資金控管參數] =================
    max_entries: int = 2                       # 大幅放寬至 5 次，改打網格化分批游擊戰 (原: 2)
    max_gap_ratio: float = 0.08                # 跳空容忍度微調 (原: 0.07)

    strong_buy_threshold: float = 0.52         # 強烈買進門檻下降至 49% (原: 0.54)
    conservative_buy_threshold: float = 0.49   # 保守買進門檻微調至 48% (原: 0.50)

    strong_buy_capital_ratio: float = 1.0      # 一旦觸發強烈買進，直接 100% 滿倉重壓 (原: 0.75)
    conservative_buy_capital_ratio: float = 0.5# 試水溫的資金比例提高至 50% (原: 0.30)

    # ================= [大盤防禦雷達門檻] =================
    safe_threshold: float = 0.54               # 大盤安全度降回中立的 50% (原: 0.54)
    cooldown_days: int = 2                     # 交易冷卻天數維持 2 天 (原: 2)

    max_return_5d: float = 0.16                # 5日漲幅過熱門檻極度收緊，超過 16% 絕不追高 (原: 0.17)
    max_bias_20: float = 0.11                  # 20日乖離率門檻大幅收緊，拒絕買進偏離均線太遠的股票 (原: 0.23)

    # ================= [動態水位風控參數] =================
    # 買進懲罰：重倉時的買進懲罰極高，強迫模型在倉位重時「冷靜」 (原: heavy=0.20, light=0.09)
    buy_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.30, light=0.15))
    # 賣出敏感度：大幅降低重倉時的賣出敏感度，搭配 100% 停損/預警機制使用 (原: heavy=0.17, light=0.06)
    sell_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.05, light=0.10))

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
                stop_loss_tolerance=-0.19,         # 容忍 19% 虧損 (原: -0.18)
                trailing_stop_drawdown=-0.16,      # 高檔回落 16% 就跑，比以前更敏銳保住獲利 (原: -0.20)
                take_profit_target=0.24,           # 賺 24% 開始分批停利 (原: 0.33)
                take_profit_sell_ratio=0.3,        # 停利時只賣 30%，剩下 70% 繼續凹大波段 (原: 0.5)
                stop_loss_sell_ratio=0.8,          # 停損時賣 80% 留點火種 (原: 0.8)
                sell_signal_threshold=0.31,        # AI 預警門檻，低於 31% 勝率準備撤退 (原: 0.32)
                warning_sell_ratio=0.7,            # AI 預警時大幅度減碼 70% 避險 (原: 0.30)

                # ================= [進攻參數] =================
                max_entries=3,                     # 真正的激進就是「一擊必殺」，不分批加碼！ (原: 4)
                max_gap_ratio=0.10,                # 無視跳空風險，容忍 10% 缺口 (原: 0.10)

                strong_buy_threshold=0.49,         # 勝率 48% 就啟動重壓模式 (原: 0.50)
                conservative_buy_threshold=0.46,   # 達標 47% 即可試單 (原: 0.45)

                strong_buy_capital_ratio=1.0,      # 一旦確認強勢，直接 100% 滿倉重擊！ (原: 0.75)
                conservative_buy_capital_ratio=0.6,# 就算是保守買進，也直接下 70% 資金 (原: 0.30)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.53,               # 大盤安全度要高達 0.53 才肯拔刀出鞘 (原: 0.52)
                cooldown_days=1,                   # 停損後隔天馬上可以再戰 (原: 1)
                max_return_5d=0.32,                # 5日漲幅高達 32% 以內都還敢追 (原: 0.22)
                max_bias_20=0.28,                  # 20日乖離高達 28% 也不怕，強勢股照追 (原: 0.14)

                # ================= [動態風控水位參數] =================
                buy_risk=RiskWeights(heavy=0.25, light=0.09),  # 買進懲罰微調 (原: heavy=0.30, light=0.14)
                sell_risk=RiskWeights(heavy=0.14, light=0.09), # 賣出敏感度提高，避免滿倉重傷 (原: heavy=0.05, light=0.03)

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
                stop_loss_tolerance=-0.15,         # 跌 3% 就強制停損，幾乎不給凹單空間 (原: -0.15)
                trailing_stop_drawdown=-0.08,      # 移動停損也收緊到 6%，保本至上 (原: -0.08)
                take_profit_target=0.15,           # 只要賺 15% 就滿足，開始停利 (原: 0.15)
                take_profit_sell_ratio=0.30,       # 停利時改為賣 30%，保留部位跟隨趨勢 (原: 0.75)
                stop_loss_sell_ratio=1.0,          # 停損時直接 100% 砍倉，不留一絲火種 (原: 0.75)
                sell_signal_threshold=0.44,        # AI 勝率低於 44% 就拉警報 (原: 0.44)
                warning_sell_ratio=1.0,            # 一收到預警，直接 100% 清倉逃命 (原: 0.30)

                # ================= [進攻參數] =================
                max_entries=4,                     # 保守反而需要「分批試單」，放寬至 4 次以分散風險 (原: 1)
                max_gap_ratio=0.09,                # 跳空容忍度維持 9% (原: 0.09)

                strong_buy_threshold=0.54,         # 要求勝率高達 62% 才敢重倉 (原: 0.58)
                strong_buy_capital_ratio=1.0,      # 一旦達標 62% 勝率，直接 100% 押滿 (原: 0.80)

                conservative_buy_threshold=0.50,   # 要求 52% 勝率才肯試水溫 (原: 0.555)
                conservative_buy_capital_ratio=0.4,# 試水溫動用 50% 資金 (原: 0.5)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.54,               # 大盤安全度降回 46% 即可接受 (原: 0.55)
                cooldown_days=5,                   # 停損後強制冷卻 5 天，避免連續吃鱉 (原: 4)
                max_return_5d=0.23,                # 5日漲幅門檻放寬 (原: 0.30)
                max_bias_20=0.27,                  # 放寬乖離率限制 (原: 0.14)

                # ================= [動態風控水位參數] =================
                # 買進風險權重 (原: heavy=0.20, light=0.15)
                buy_risk=RiskWeights(heavy=0.15, light=0.06),
                # 賣出風險權重 (原: heavy=0.18, light=0.05)
                sell_risk=RiskWeights(heavy=0.06, light=0.05),

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