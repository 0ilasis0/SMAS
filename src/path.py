import sys
from dataclasses import dataclass
from pathlib import Path


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
    IDSS_DATA = PathBase.processed / "idss_data.db"
    RESULP_REPORT = PathBase.processed / "report"
    GEMINI_KEY = PathBase.raw / ".env"
    XGB_MODEL = PathBase.model / "xgb_model.json"

    @classmethod
    def get_backtest_report_path(cls, ticker: str) -> Path:
        """
        根據 ticker 動態生成回測報告的 CSV 路徑。
        例如: '006208.TW' -> output/006208_backtest.csv
        """
        # 確保輸出目錄存在
        cls.RESULP_REPORT.mkdir(parents=True, exist_ok=True)
        clean_ticker = ticker.replace(".TW", "")

        # 組合出最終的路徑
        return cls.RESULP_REPORT / f"{clean_ticker}_backtest.csv"
