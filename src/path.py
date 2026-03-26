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
    MODEL_DIR = PathBase.model

    @classmethod
    def get_backtest_report_path(cls, ticker: str) -> Path:
        return cls._generate_dynamic_path(ticker, cls.RESULT_REPORT, "_backtest", ".csv")

    @classmethod
    def get_xgboost_model_path(cls, ticker: str) -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, "_xgb_model", ".json")

    @classmethod
    def get_dl_model_path(cls, ticker: str, rnn_type: "RNNType") -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{rnn_type.name}_model", ".pth")

    @classmethod
    def get_dl_scalar_path(cls, ticker: str, rnn_type: "RNNType") -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{rnn_type.name}_dl_scaler", ".joblib")

    @classmethod
    def get_meta_model_path(cls, ticker: str) -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, "_meta_model", ".joblib")


    @classmethod
    def _generate_dynamic_path(cls, ticker: str, base_dir: Path, suffix: str, ext: str) -> Path:
        """
        通用路徑生成邏輯 (通式)
        """
        # 確保目錄存在
        base_dir.mkdir(parents=True, exist_ok=True)
        # 統一處理 Ticker
        clean_ticker = ticker.split('.')[0]
        # 組合最終路徑
        return base_dir / f"{clean_ticker}{suffix}{ext}"
