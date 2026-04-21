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
    stop_loss_tolerance: float = -0.09           # 強制停損容忍度，稍微收緊 (原: -0.10)
    trailing_stop_drawdown: float = -0.12        # 移動停損回落容忍度，放寬以防洗盤 (原: -0.10)
    stop_loss_sell_ratio: float = 0.8            # 觸發停損時的賣出比例維持不變 (原: 0.80)

    sell_signal_threshold: float = 0.28          # AI 勝率低迷預警門檻 (原: 0.27)
    warning_sell_ratio: float = 0.3              # AI 預警時的減碼比例大幅下降，不再輕易下車 (原: 0.70)

    take_profit_target: float = 0.20             # 極端停利目標拉高至 20% (原: 0.18)
    take_profit_sell_ratio: float = 0.3          # 觸發停利時只賣 30%，讓利潤奔跑 (原: 1.0)

    # ================= 進攻與資金控管參數 =================
    max_entries: int = 2                         # 單一波段最大加碼次數恢復為 2 (原: 1)
    max_gap_ratio: float = 0.07                  # 跳空容忍度放寬至 7% (原: 0.05)

    strong_buy_threshold: float = 0.54           # 強烈買進訊號門檻大幅降低 (原: 0.55)
    strong_buy_capital_ratio: float = 0.75        # 只要達標，直接 100% 資金重壓 (原: 0.70)

    conservative_buy_threshold: float = 0.50     # 保守買進訊號門檻 (原: 0.50)
    conservative_buy_capital_ratio: float = 0.3 # 試水溫維持動用 30% 資金 (原: 0.40)

    # ================= 大盤防禦雷達門檻 =================
    safe_threshold: float = 0.52                 # 大盤安全度門檻稍微提高 (原: 0.45)
    cooldown_days: int = 2                       # 交易冷卻天數縮短 (原: 3)

    max_return_5d: float = 0.17                  # 5日漲幅過熱門檻極度收緊，絕不追高 (原: 0.23)
    max_bias_20: float = 0.23                    # 20日乖離率過熱門檻微調 (原: 0.20)

    # ================= 動態水位風控參數 =================
    # 買進風險懲罰：倉位重時的買進懲罰加重至 0.20，防禦度極高 (原: heavy=0.15, light=0.10)
    buy_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.20, light=0.09))
    # 賣出風險敏感：倉位重時的賣出敏感度提高至 0.17 (原: heavy=0.10, light=0.05)
    sell_risk: RiskWeights = field(default_factory=lambda: RiskWeights(heavy=0.17, light=0.06))

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
                stop_loss_tolerance=-0.18,         # 容忍 18% 虧損 (原: -0.20)
                trailing_stop_drawdown=-0.20,      # 回落 20% 才跑，給予極端震盪空間 (原: -0.18)
                take_profit_target=0.33,           # 賺 33% 才開始停利，超級讓利潤奔跑 (原: 0.25)
                take_profit_sell_ratio=0.5,        # 停利時只賣一半，剩下的繼續凹 (原: 1.0)
                stop_loss_sell_ratio=0.8,          # 停損時賣 80% 留點火種 (原: 1.0)
                sell_signal_threshold=0.32,        # AI 預警門檻維持不變 (原: 0.32)
                warning_sell_ratio=0.30,           # AI 預警時只小幅度減碼 30% (原: 0.70)

                # ================= [進攻參數] =================
                max_entries=4,                     # ⚠️ AI 判定激進型不該分批加碼，直接單次定勝負 (原: 3)
                max_gap_ratio=0.10,                # 完全無視跳空風險，容忍 10% 缺口 (原: 0.02)

                strong_buy_threshold=0.50,         # 勝率 46% 就敢重壓 (原: 0.50)
                conservative_buy_threshold=0.45,   # (原: 0.48)

                strong_buy_capital_ratio=0.75,     # 重壓 80% 資金 (原: 0.75)
                conservative_buy_capital_ratio=0.3, # (原: 0.30)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.49,               # 激進型反而要求大盤要過半安全 (0.51) 才肯出手 (原: 0.44)
                cooldown_days=1,                   # 停損後隔天馬上可以再戰 (原: 3)
                max_return_5d=0.22,                # 5日漲幅超過 22% 就不追高 (原: 0.30)
                max_bias_20=0.14,                  # 20日乖離超過 14% 就不追高 (原: 0.20)

                # ================= [動態風控水位參數] =================
                # 買進懲罰大幅提高，代表 AI 不希望在倉位重的時候繼續無腦買
                buy_risk=RiskWeights(heavy=0.30, light=0.14), # (原: heavy=0.10, light=0.05)
                # 賣出敏感度降低，代表 AI 傾向抱住獲利，不受一點風吹草動就賣出
                sell_risk=RiskWeights(heavy=0.05, light=0.03),# (原: heavy=0.11, light=0.10)

                # ================= [智慧定價參數 (手動設定)] =================
                buy_panic_discount_atr=0.5,      # 別人恐慌接刀要折價 1.2，激進型只要折價 0.5 就敢搶反彈
                buy_strong_discount_atr=0.0,     # 勝率高時，直接掛「平盤價 (0 折價)」全力追擊
                buy_normal_discount_atr=0.3,     # 常規震盪也只等微幅拉回 0.3 ATR 就買
                sell_strong_premium_atr=0.3,     # 賣出時不想等太久，溢價 0.3 ATR 就願意賣給別人
                sell_normal_premium_atr=0.1,     # 常規轉弱時，幾乎平盤就趕快逃命
                pricing_buy_strong_prob=0.58,    # 只要勝率 58% 就認定是強勢股，啟動追價模式

                # ================= [LLM 參數保留手動設定] =================
                min_sentiment_score=4,
            )

        elif persona == TradingPersona.CONSERVATIVE:
            # 🛡️ 保守型：草木皆兵，極度要求大盤環境安全
            return StrategyConfig(
                # ================= [防守參數] =================
                stop_loss_tolerance=-0.15,         # 跌 5% 就強制停損 (原: -0.08)
                trailing_stop_drawdown=-0.08,      # ⚠️ 極度神經質：只要回落 1% 就立刻停損逃命 (原: -0.08)
                take_profit_target=0.15,           # 只要賺 8% 就滿足，開始停利 (原: 0.15)
                take_profit_sell_ratio=0.75,       # 停利時一次賣掉 70% 鎖住獲利 (原: 0.75)
                stop_loss_sell_ratio=0.75,         # 停損時賣掉 80% (原: 0.75)
                sell_signal_threshold=0.44,        # AI 勝率低於 37% 就預警 (原: 0.44)
                warning_sell_ratio=0.30,           # 預警時先微幅減碼 30% 觀察 (原: 0.75)

                # ================= [進攻參數] =================
                max_entries=1,                     # 絕不攤平加碼，買一次定生死 (原: 1)
                max_gap_ratio=0.09,                # 意外地放寬了跳空容忍度至 9% (原: 0.03)

                strong_buy_threshold=0.58,         # 要求勝率高達 58% 才敢重倉 (原: 0.59)
                strong_buy_capital_ratio=0.80,     # 重倉只敢動用 80% 資金，絕對不 All-in (原: 1.0)

                conservative_buy_threshold=0.555,   # 要求 56% 勝率才肯試水溫 (原: 0.54)
                conservative_buy_capital_ratio=0.5,# 試水溫動用 50% 資金 (原: 0.40)

                # ================= [大盤防禦參數] =================
                safe_threshold=0.54,               # ⚠️ 大盤安全度必須高達 54% 才肯出手 (原: 0.47)
                cooldown_days=4,                   # 停損後冷卻期拉長到 4 天 (原: 3)
                max_return_5d=0.30,                # 5日漲幅門檻放寬 (原: 0.16)
                max_bias_20=0.14,                  # ⚠️ 嚴格限制：20日乖離超過 14% 絕對不追高 (原: 0.23)

                # ================= [動態風控水位參數] =================
                # 買進風險權重 (原: heavy=0.10, light=0.14)
                buy_risk=RiskWeights(heavy=0.20, light=0.15),  # 倉位重時極度懲罰買進訊號
                # 賣出風險權重 (原: heavy=0.11, light=0.09)
                sell_risk=RiskWeights(heavy=0.18, light=0.05), # 倉位重時對賣出訊號極度敏感

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