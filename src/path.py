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
    raw = resource_path("data", "raw")

@dataclass(frozen = True)
class PathConfig:
    IDSS_DATA = PathBase.processed / "idss_data.db"
    GEMINI_KEY = PathBase.raw / ".env"
