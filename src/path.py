import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.const import DLModelType, RNNType

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
    EXPERIMENT_RESULTS = PathBase.processed / "report" / "experiment_results.csv"
    WATCHLIST = PathBase.processed / "watchlist.json"
    SETTINGS = PathBase.processed / "settings.json"
    PORTFOLIO = PathBase.processed / "portfolio.json"
    IDSS_DATA = PathBase.processed / "idss_data.db"
    LLM_CACHE = PathBase.processed / "llm_cache.db"
    CACHE_FILE = PathBase.processed / "update_cache.json"
    GEMINI_KEY = PathBase.raw / ".env"
    MODEL_DIR = PathBase.model

    @classmethod
    def get_backtest_report_path(cls, ticker: str) -> Path:
        return cls._generate_dynamic_path(ticker, cls.RESULT_REPORT, "_backtest", ".csv")

    @classmethod
    def get_chart_report_path(cls, ticker: str) -> Path:
        return cls._generate_dynamic_path(ticker, cls.RESULT_REPORT, f"_chart", ".png")

    @classmethod
    def get_xgboost_model_path(cls, ticker: str, oos_days: int = 0) -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, "_xgb_model", ".json", oos_days)

    @classmethod
    def get_dl_model_path(cls, ticker: str, dl_type: "DLModelType", rnn_type: "RNNType", oos_days: int = 0) -> Path:
        dl_name = dl_type.name if dl_type else "DL"
        rnn_name = rnn_type.name if rnn_type else ""
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{dl_name}_{rnn_name}_model", ".pth", oos_days)

    @classmethod
    def get_dl_scalar_path(cls, ticker: str, dl_type: "DLModelType", rnn_type: "RNNType", oos_days: int = 0) -> Path:
        dl_name = dl_type.name if dl_type else "DL"
        rnn_name = rnn_type.name if rnn_type else ""
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{dl_name}_{rnn_name}_scaler", ".joblib", oos_days)

    @classmethod
    def get_meta_model_path(cls, ticker: str, oos_days: int = 0) -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, "_meta_model", ".joblib", oos_days)

    @classmethod
    def get_market_model_path(cls, oos_days: int = 0) -> Path:
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        oos_suffix = f"_oos_{oos_days}" if oos_days > 0 else ""
        return cls.MODEL_DIR / f"universal_market_model{oos_suffix}.joblib"

    @classmethod
    def _generate_dynamic_path(cls, ticker: str, base_dir: Path, suffix: str, ext: str, oos_days: int = 0) -> Path:
        """通用路徑生成邏輯 """
        base_dir.mkdir(parents=True, exist_ok=True)
        clean_ticker = ticker.split('.')[0]
        oos_suffix = f"_oos_{oos_days}" if oos_days > 0 else ""
        return base_dir / f"{clean_ticker}{suffix}{oos_suffix}{ext}"
