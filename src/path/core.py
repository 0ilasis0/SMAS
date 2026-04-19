import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from debug import dbg

if TYPE_CHECKING:
    from ml.const import DLModelType, RNNType

def _resource_path(*paths):
    """
    取得外部資源路徑：
    - 打包成 exe 時：使用 exe 同目錄
    - 開發模式：使用專案根目錄
    """
    if getattr(sys, "frozen", False):
        # exe 打包後使用的路徑
        base_path = Path(sys.executable).resolve().parent.parent
    else:
        # 開發環境使用的路徑
        base_path = Path(__file__).resolve().parent.parent.parent

    return base_path.joinpath(*paths)


@dataclass(frozen = True)
class _PathBase:
    processed = _resource_path("data", "processed")
    model = _resource_path("data", "processed", "model")
    report = _resource_path("data", "processed", "report")
    raw = _resource_path("data", "raw")

    @classmethod
    def get_all_paths(cls):
        return [v for _, v in vars(cls).items() if isinstance(v, Path)]

@dataclass(frozen = True)
class PathConfig:
    MODEL_DIR = _PathBase.model
    RESULT_REPORT = _PathBase.report
    EXPERIMENT_DETAILS = _PathBase.report / "experiment_detail.csv"
    EXPERIMENT_SUMMARY = _PathBase.report / "experiment_summary.csv"
    ALL_STOCKS_PERSONA = _PathBase.report / "all_stocks_persona.csv"
    SUMMARY_PERSONA = _PathBase.report/ "summary_persona.csv"
    SETTINGS = _PathBase.processed / "settings.json"
    PORTFOLIO = _PathBase.processed / "portfolio.json"
    CACHE_FILE = _PathBase.processed / "update_cache.json"
    IDSS_DATA = _PathBase.processed / "idss_data.db"
    LLM_CACHE = _PathBase.processed / "llm_cache.db"
    GEMINI_KEY = _PathBase.raw / "key.env"

    @classmethod
    def get_all(cls):
        return [e for e in cls]

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
        rnn_name = f"{rnn_type.name}_" if rnn_type else ""
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{dl_name}_{rnn_name}model", ".pth", oos_days)

    @classmethod
    def get_dl_scalar_path(cls, ticker: str, dl_type: "DLModelType", rnn_type: "RNNType", oos_days: int = 0) -> Path:
        dl_name = dl_type.name if dl_type else "DL"
        rnn_name = f"{rnn_type.name}_" if rnn_type else ""
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, f"_{dl_name}_{rnn_name}scaler", ".joblib", oos_days)

    @classmethod
    def get_meta_model_path(cls, ticker: str, oos_days: int = 0) -> Path:
        return cls._generate_dynamic_path(ticker, cls.MODEL_DIR, "_meta_model", ".joblib", oos_days)

    @classmethod
    def get_market_model_path(cls, oos_days: int = 0) -> Path:
        if not cls.MODEL_DIR.exists():
            cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            dbg.log(f"建立新資料夾: {cls.MODEL_DIR}")

        oos_suffix = f"_oos_{oos_days}" if oos_days > 0 else ""
        return cls.MODEL_DIR / f"universal_market_model{oos_suffix}.joblib"

    @classmethod
    def _generate_dynamic_path(cls, ticker: str, base_dir: Path, suffix: str, ext: str, oos_days: int = 0) -> Path:
        """通用路徑生成邏輯 """
        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
            dbg.log(f"建立新資料夾: {base_dir}")

        clean_ticker = ticker.split('.')[0]
        oos_suffix = f"_oos_{oos_days}" if oos_days > 0 else ""
        return base_dir / f"{clean_ticker}{suffix}{oos_suffix}{ext}"


def setup_filesystem():
    """
    確保所有靜態路徑的「資料夾」都存在。
    """
    try:
        for d in _PathBase.get_all_paths():
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                dbg.log(f"[系統初始化] 建立新資料夾: {d}")

    except Exception as e:
        dbg.error(f"路徑系統初始化警告: {e}")
