import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.params import RNNType

def resource_path(*paths):
    """
    取得外部資源路徑：
    - 打包成 exe 時：使用 exe 同目錄
    - 開發模式：使用專案根目錄
    """
    if getattr(sys, "frozen", False):
        # exe 打包後使用的路徑
        base_path = Path(sys.executable).resolve().parent
    else:
        # 開發環境使用的路徑
        base_path = Path(__file__).resolve().parent.parent

    return base_path.joinpath(*paths)


@dataclass(frozen = True)
class PathBase:
    processed = resource_path("data", "processed")
    model = resource_path("data", "processed", "model")
    raw = resource_path("data", "raw")

@dataclass(frozen = True)
class PathConfig:
    RESULT_REPORT = PathBase.processed / "report"
    IDSS_DATA = PathBase.processed / "idss_data.db"
    GEMINI_KEY = PathBase.raw / ".env"
    XGB_MODEL = PathBase.model / "xgb_model.json"
    MODEL_DIR = PathBase.model

    @classmethod
    def get_backtest_report_path(cls, ticker: str) -> Path:
        """
        根據 ticker 動態生成回測報告的 CSV 路徑。
        例如: '006208.TW' -> output/006208_backtest.csv
        """
        # 確保輸出目錄存在
        cls.RESULT_REPORT.mkdir(parents=True, exist_ok=True)
        clean_ticker = ticker.replace(".TW", "")

        # 組合出最終的路徑
        return cls.RESULT_REPORT / f"{clean_ticker}_backtest.csv"

    @classmethod
    def get_dl_model_path(cls, ticker: str, rnn_type: "RNNType") -> Path:
        """
        根據 ticker 與 RNN 類型動態生成權重檔路徑。
        例如: ('006208.TW', LSTM) -> model/006208_LSTM_model.pth
        """
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        clean_ticker = ticker.replace(".TW", "")

        return cls.MODEL_DIR / f"{clean_ticker}_{rnn_type.name}_model.pth"
