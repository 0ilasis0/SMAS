from debug import dbg
from ml.engine import QuantAIEngine
from ml.params import SessionConfig

if __name__ == "__main__":
    # 1. 啟動引擎
    c_ticker = "2388.TW"
    engine = QuantAIEngine(ticker=c_ticker)
    dbg.log(f"current {c_ticker} mode is {SessionConfig.rnn_type}")

    # 2. 假設 UI 按下了「更新資料庫」按鈕
    # engine.update_market_data()

    # 3. 假設 UI 按下了「重新訓練 AI 模型」按鈕
    # 傳入 save_models=True，它就會在 CV 驗證完後，自動幫你把 .pth / .json 存下來
    engine.train_all_models(save_models=True)

    # 4. 假設 UI 啟動，準備進入每日自動交易監控模式
    # engine.load_inference_models()