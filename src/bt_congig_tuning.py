import numpy as np
import optuna
from tqdm import tqdm

from bt.backtest import BacktestEngine
from bt.strategy_config import RiskWeights, StrategyConfig
from debug import dbg
from ml.engine import QuantAIEngine
from path import PathConfig

dbg.toggle()

def fetch_data_for_optuna(tickers: list[str], oos_days: int = 240) -> dict:
    """預先抓好所有股票的測試資料，放進記憶體，避免尋優過程中重複讀取"""
    data_dict = {}
    print("📥 正在預載回測資料，請稍候...")
    for ticker in tickers:
        try:
            engine = QuantAIEngine(ticker=ticker, oos_days=oos_days)
            if engine.load_inference_models():
                df = engine.generate_backtest_data()
                if not df.empty:
                    data_dict[ticker] = df.tail(oos_days)
        except Exception as e:
            print(f"⚠️ {ticker} 載入失敗: {e}")

    print(f"✅ 成功載入 {len(data_dict)} 檔股票資料！")
    return data_dict

def objective(trial, data_dict: dict, initial_cash: float, persona_mode: str):
    """
    Optuna 的目標函數：根據不同性格 (persona_mode) 給予不同的評分權重
    """
    # ================= 1. 讓 AI 動態選擇參數 (依據性格切換邊界) =================
    if persona_mode == "aggressive":
        # 激進型：容忍深跌、買進門檻極低、無視大盤
        sl_bounds = (-0.20, -0.08)
        buy_bounds = (0.40, 0.55)
        safe_bounds = (0.20, 0.40)
        profit_bounds = (0.15, 0.30)

    elif persona_mode == "conservative":
        # 保守型：嚴格停損、要求極高勝率、大盤必須安全
        sl_bounds = (-0.08, -0.02)
        buy_bounds = (0.55, 0.65)
        safe_bounds = (0.45, 0.70)
        profit_bounds = (0.03, 0.10)

    else:
        # 穩健型 (MODERATE)
        sl_bounds = (-0.12, -0.05)
        buy_bounds = (0.45, 0.60)
        safe_bounds = (0.30, 0.50)
        profit_bounds = (0.08, 0.20)

    # [防守參數]
    stop_loss_tolerance = trial.suggest_float('stop_loss_tolerance', sl_bounds[0], sl_bounds[1], step=0.01)

    trailing_upper_bound = min(sl_bounds[1] + 0.03, -0.01)
    trailing_stop_drawdown = trial.suggest_float('trailing_stop_drawdown', sl_bounds[0], trailing_upper_bound, step=0.01)
    take_profit_target = trial.suggest_float('take_profit_target', profit_bounds[0], profit_bounds[1], step=0.01)

    take_profit_sell_ratio = trial.suggest_categorical('take_profit_sell_ratio', [0.3, 0.5, 0.7, 1.0])
    warning_sell_ratio = trial.suggest_categorical('warning_sell_ratio', [0.3, 0.5, 0.7, 1.0])
    stop_loss_sell_ratio = trial.suggest_categorical('stop_loss_sell_ratio', [0.8, 1.0])

    sell_signal_threshold = trial.suggest_float('sell_signal_threshold', 0.25, 0.45, step=0.01)

    # [進攻參數]
    max_entries = trial.suggest_int('max_entries', 1, 5)
    max_gap_ratio = trial.suggest_float('max_gap_ratio', 0.02, 0.10, step=0.01)

    t1 = trial.suggest_float('buy_threshold_1', buy_bounds[0], buy_bounds[1], step=0.01)
    t2 = trial.suggest_float('buy_threshold_2', buy_bounds[0], buy_bounds[1], step=0.01)
    strong_buy_threshold = max(t1, t2)
    conservative_buy_threshold = min(t1, t2)

    conservative_buy_capital_ratio = trial.suggest_categorical('conservative_buy_capital_ratio', [0.3, 0.5, 0.7])
    strong_buy_capital_ratio = trial.suggest_categorical('strong_buy_capital_ratio', [0.8, 1.0])

    # [大盤防禦參數]
    safe_threshold = trial.suggest_float('safe_threshold', safe_bounds[0], safe_bounds[1], step=0.01)
    cooldown_days = trial.suggest_int('cooldown_days', 1, 5)
    max_return_5d = trial.suggest_float('max_return_5d', 0.15, 0.40, step=0.01)
    max_bias_20 = trial.suggest_float('max_bias_20', 0.10, 0.30, step=0.01)

    # [動態風控水位參數]
    buy_heavy = trial.suggest_float('buy_heavy', 0.1, 0.3, step=0.05)
    buy_light = trial.suggest_float('buy_light', 0.05, 0.15, step=0.01)
    sell_heavy = trial.suggest_float('sell_heavy', 0.05, 0.2, step=0.01)
    sell_light = trial.suggest_float('sell_light', 0.01, 0.1, step=0.01)

    # ================= 2. 建立策略設定 (保持不變) =================
    config = StrategyConfig(
        stop_loss_tolerance=stop_loss_tolerance,
        trailing_stop_drawdown=trailing_stop_drawdown,
        stop_loss_sell_ratio=stop_loss_sell_ratio,
        take_profit_target=take_profit_target,
        take_profit_sell_ratio=take_profit_sell_ratio,
        sell_signal_threshold=sell_signal_threshold,
        warning_sell_ratio=warning_sell_ratio,
        max_entries=max_entries,
        max_gap_ratio=max_gap_ratio,
        strong_buy_threshold=strong_buy_threshold,
        strong_buy_capital_ratio=strong_buy_capital_ratio,
        conservative_buy_threshold=conservative_buy_threshold,
        conservative_buy_capital_ratio=conservative_buy_capital_ratio,
        safe_threshold=safe_threshold,
        cooldown_days=cooldown_days,
        max_return_5d=max_return_5d,
        max_bias_20=max_bias_20,
        buy_risk=RiskWeights(heavy=buy_heavy, light=buy_light),
        sell_risk=RiskWeights(heavy=sell_heavy, light=sell_light),
        enable_llm_oracle=False
    )

    # ================= 3. 多檔股票聯合評測 =================
    sharpes = []
    returns = []
    mdds = [] # 🌟 新增：收集最大回撤

    for ticker, df in data_dict.items():
        engine = BacktestEngine(initial_cash=initial_cash, ticker=ticker, strategy=config)
        stats = engine.run(df, silence=True)

        if stats and (stats['buy_count'] + stats['sell_count']) > 2:
            sharpes.append(stats['sharpe'])
            returns.append(stats['total_return'])
            mdds.append(stats['mdd']) # 收集 MDD
        else:
            sharpes.append(-2.0)
            returns.append(-1.0)
            mdds.append(-1.0) # 極端懲罰

    # ================= 4. 定義最終分數 (根據不同性格) =================
    avg_sharpe = np.mean(sharpes)
    avg_return = np.mean(returns)
    avg_mdd = np.mean(mdds)

    if persona_mode == "aggressive":
        # 🔥 激進型：追求絕對報酬，給予報酬率極高權重，甚至稍微容忍回撤
        final_score = avg_return

    elif persona_mode == "conservative":
        # 🛡️ 保守型：極度厭惡風險。若平均回撤大於 10% (-0.1)，直接淘汰
        if avg_mdd < -0.1:
            return -999.0
        # 如果活下來了，主要看夏普值 (穩定性)
        final_score = avg_sharpe

    else:
        # ⚖️ 穩健型 (moderate)：兼顧風險與報酬
        final_score = avg_sharpe + (avg_return * 2)

    return final_score

def run_optimization():
    # 🌟 這裡設定您這次想要找哪一種性格！
    # 可以填入: "aggressive", "moderate", 或 "conservative"
    TARGET_PERSONA = "conservative"

    print("="*60)
    print(f"🚀 IDSS 全維度尋優引擎啟動 | 目標性格: [{TARGET_PERSONA.upper()}]")
    print("="*60)

    test_tickers = [
        "2330.TW", "0050.TW", "2603.TW", "2317.TW",
        "2881.TW", "2409.TW", "2388.TW"
    ]

    data_dict = fetch_data_for_optuna(test_tickers, oos_days=240)
    if not data_dict:
        return

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    PathConfig.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
    db_path = PathConfig.RESULT_REPORT / "idss_optuna_study.db"
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"

    print(f"📁 尋優資料庫連結至: {db_path.name}")

    # 🌟 根據選擇的性格，建立不同的 Study Name，這樣資料庫才不會混在一起
    study_name = f"IDSS_{TARGET_PERSONA.capitalize()}_Baseline"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=db_url,
        load_if_exists=True
    )

    TARGET_TOTAL_TRIALS = 2000
    initial_cash = 2_000_000

    completed_trials = len(study.trials)
    remaining_trials = max(0, TARGET_TOTAL_TRIALS - completed_trials)

    if remaining_trials == 0:
        print(f"✅ [{TARGET_PERSONA.upper()}] 尋優專案已完成 {TARGET_TOTAL_TRIALS} 次測試！")
    else:
        print(f"⏳ 目前已完成 {completed_trials} 次，剩餘 {remaining_trials} 次測試即將開始...")

        with tqdm(total=remaining_trials, desc="🎯 尋優進度", unit="trial") as pbar:

            def update_tqdm_callback(study, trial):
                pbar.set_postfix({"Best Score": f"{study.best_value:.4f}"})
                pbar.update(1)

            # 🌟 記得把 persona_mode 傳給 objective 函數
            study.optimize(
                lambda trial: objective(trial, data_dict, initial_cash=initial_cash, persona_mode=TARGET_PERSONA),
                n_trials=remaining_trials,
                callbacks=[update_tqdm_callback]
            )

    # ================= 輸出最終結果 =================
    print("\n\n" + "="*60)
    print(f"🏆 【尋優完成】最強 {TARGET_PERSONA.upper()} 參數誕生！")
    print("="*60)

    if len(study.trials) > 0 and study.best_trial:
        print(f"🥇 最高綜合評分: {study.best_value:.4f} (在第 {study.best_trial.number} 次尋優找到)")
        print(f"\n📝 請將以下參數寫入 PersonaFactory ({TARGET_PERSONA.upper()})：")

        best_params = study.best_params

        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    else:
        print("⚠️ 尚未產生任何有效的尋優結果。")

    print("="*60)

if __name__ == "__main__":
    run_optimization()
