from enum import Enum, auto

from py.path.manager import JsonFileID


class SaveID(Enum):
    # --- 系統設定 ---
    SYS_SONG = auto()
    SYS_VOLUME = auto()
    SYS_SCALE  = auto()

    # 單人關卡破關進度
    SINGLE_MENU_GRID = auto()

    # --- 排行榜 ---
    RANK_1ST = auto()
    RANK_2ND = auto()
    RANK_3RD = auto()



class SaveJson:
    """
    建立 SaveID 到 JSON 路徑的映射
    Value: (JsonID, Key_L1, Key_L2, ... Key_Ln)
    """
    mapping = {
        # --- 系統設定 (單層) ---
        SaveID.SYS_SONG:   (JsonFileID.SAVE.value, "SYS_CONFIG", "sys_select_song"),
        SaveID.SYS_VOLUME: (JsonFileID.SAVE.value, "SYS_CONFIG", "sys_volume"),
        SaveID.SYS_SCALE:  (JsonFileID.SAVE.value, "SYS_CONFIG", "sys_window_scale"),

        # --- 單人關卡破關進度 ---
        SaveID.SINGLE_MENU_GRID: (JsonFileID.SAVE.value, "SINGLE_MENU", "level_grid"),

        # --- 排行榜 (多層結構：RANK -> display -> rank_x) ---
        SaveID.RANK_1ST:   (JsonFileID.SAVE.value, "RANK", "display", "rank_0"),
        SaveID.RANK_2ND:   (JsonFileID.SAVE.value, "RANK", "display", "rank_1"),
        SaveID.RANK_3RD:   (JsonFileID.SAVE.value, "RANK", "display", "rank_2"),
    }
