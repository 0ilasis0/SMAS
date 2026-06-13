import traceback

from data.manager import DataManager
from data.updater import DataUpdater
from debug import dbg


def sync_market_data(ticker: str, force_wipe: bool = False, force_sync: bool = False) -> bool:
    """
    UI 專屬的資料同步管線：負責調度後端引擎抓取個股、大盤與企業事件
    force_wipe：本地端舊資料要不要刪除
    force_sync：今天明明已經抓過資料了，還要不要強行去網路再抓一次（覆蓋）
    """
    dbg.log(f"[{ticker}] 啟動獨立資料同步管線 (Wipe={force_wipe}, Sync={force_sync})...")

    try:
        # 實體化後端的資料管理器與更新器
        data_mg = DataManager()
        updater = DataUpdater(data_mg)

        success = updater.update_market_data(ticker=ticker, force_wipe=force_wipe, force_sync=force_sync)

        if success:
            dbg.log(f"[{ticker}] 資料同步成功！資料庫已為最新狀態。")
        else:
            dbg.error(f"[{ticker}] 資料同步失敗，請檢查網路連線或 API 流量限制。")

        return success

    except Exception as e:
        error_details = traceback.format_exc()
        dbg.error(f"[{ticker}] 資料同步過程中發生未預期崩潰: {e}\n追蹤:\n{error_details}")
        return False