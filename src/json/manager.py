import json

from debug import dbg
from py.path.manager import JsonFileID, PathConfig


class JsonManager:
    def __init__(self) -> None:
        self.base  = 'utf-8'
        self.json_data = {}

        # 建立 ID 到路徑的映射表
        self.path_map = {
            JsonFileID.SAVE: PathConfig.json_save,
            JsonFileID.DISPLAY: PathConfig.json_display
        }

        self._setup()

    def _setup(self):
        # 初始化json內容
        for file_id, path in self.path_map.items():
            self._load_json_file(path, file_id)

    def _load_json_file(self, file_path, file_id = None):
        """
        通用讀取方法：讀取 JSON 並以「檔名」為 Key 存入 self.json_data
        例：讀取 'data/display.json' -> self.json_data['display'] = {...}
        :param file_id: 強制指定存入的 Key (Enum)，若無則使用檔名
        """
        if not file_path.exists():
            dbg.error(f"檔案不存在：{file_path}")
            return

        try:
            with open(file_path, "r", encoding = self.base) as f:
                data = json.load(f)

            # 取得檔名 (不含副檔名)
            key = file_id if file_id else file_path.stem

            if isinstance(data, dict):
                self.json_data[key] = data
            else:
                dbg.error(f"JSON 格式錯誤：{file_path} 根節點必須是 dict")

        except Exception as e:
            dbg.error(f"讀取失敗 {file_path}: {e}")

    @staticmethod
    def _read_existing(file_path, encoding):
        """讀取舊有 JSON 檔案，若失敗則回傳空 dict"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_json(
            self,
            file_path,
            data,
            mode = "w",
            encoding = None,
            indent = 4,
            only_keys = None
        ):
        """
        將資料寫入 JSON 檔案
        - mode ='w': 覆蓋
        - mode ='a': 附加（dict → 合併，list → 延伸）
        - only_keys: 只更新指定 key，不會洗掉其他 key
        """
        if encoding is None:
            encoding = self.base

        existing_data = {}

        # 只有 append 模式 或 only_keys 指定時，才需要先讀取舊檔
        if mode == "a" or only_keys is not None:
            existing_data = self._read_existing(file_path, encoding)
            if not isinstance(existing_data, (dict, list)):
                existing_data = {}  # 非法格式強制重置

        # --- only_keys 更新邏輯 ---
        if only_keys is not None:
            if not isinstance(existing_data, dict):
                existing_data = {}
            for k in only_keys:
                if k in data:
                    existing_data[k] = data[k]
            data_to_write = existing_data
        else:
            data_to_write = data

        # --- 附加模式 ---
        if mode == "a":
            if isinstance(existing_data, dict) and isinstance(data_to_write, dict):
                existing_data.update(data_to_write)
                data_to_write = existing_data
            elif isinstance(existing_data, list) and isinstance(data_to_write, list):
                data_to_write = existing_data + data_to_write
            else:
                raise ValueError("無法附加不同型別的 JSON 資料")

        # --- 寫回檔案 ---
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data_to_write, f, ensure_ascii = False, indent = indent)

    def delete_data(self, file_id: JsonFileID, *keys):
        """
        刪除指定資料
        :param file_id: JsonFileID (例如 'save')
        :param keys: 路徑 (例如 'SYS_CONFIG', StockCol.VOLUME)
        """
        if file_id not in self.json_data:
            dbg.error(f"[JsonManager] 刪除失敗：找不到檔案命名空間 '{file_id}'")
            return False

        if not keys:
            dbg.war("[JsonManager] delete_data 需要至少一個 key")
            return False

        try:
            target_parent = self.json_data[file_id]
            current_path = [file_id]

            for key in keys[:-1]:
                current_path.append(key)
                target_parent = target_parent[key]

            # 執行刪除
            last_key = keys[-1]
            if last_key in target_parent:
                del target_parent[last_key]
                dbg.log(f"成功刪除路徑: {current_path + [last_key]}")
                return True
            else:
                dbg.error(f"刪除失敗：Key '{last_key}' 不存在於路徑 {current_path}")
                return False

        except (KeyError, TypeError) as e:
            dbg.error(f"刪除失敗：路徑錯誤 {current_path} -> {keys} ({e})")
            return False

    def get_data(self, file_id: JsonFileID, *keys, silent = False):
        """
        取得指定檔案內的 JSON 資料
        :param filename: JSON 檔名 (不含副檔名，例如 'display')
        :param keys: 內部的層級路徑
        """
        # 先檢查檔案是否存在
        if file_id not in self.json_data:
            if not silent:
                dbg.error(f"[JsonManager] 找不到檔案命名空間: '{file_id}' (請確認 setup 是否有載入)")
            return None

        data = self.json_data[file_id]

        # 遍歷內部的 Keys
        try:
            current_path = [file_id] # 用來記錄目前找的路徑，方便報錯

            for key in keys:
                current_path.append(key)
                data = data[key]

            return data

        except (KeyError, TypeError):
            if silent: return None
            # 這裡的報錯會非常清楚，告訴你是哪一個環節斷掉了
            dbg.error(f"[JsonManager] 查找失敗，路徑斷在: {current_path}")
            return None

    def update_data(
            self,
            file_id: JsonFileID,
            *keys,
            value,
            index = None
        ):
        """
        更新記憶體中的 JSON 資料
        :param file_id: JsonID (例如 'save')
        :param keys: 路徑 (例如 'SYS_CONFIG', StockCol.VOLUME)
        :param value: 要寫入的值
        :param index:若指定，則更新 List 中第 n 個位置的值；若為 None，則覆寫整個 List
        """
        # 確保檔案命名空間存在
        if file_id not in self.json_data:
            self.json_data[file_id] = {}

        target = self.json_data[file_id]

        # 導航到目標節點的上一層 (Dict 層導航)
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

            # 防呆：確保路徑暢通 (Dict)
            if not isinstance(target, dict):
                target = {} # 強制重置為 dict

        # 處理最後一層 (List 層處理)
        last_key = keys[-1]

        # 指定了 index (更新 List 中的某個值)
        if index is not None:
            # 確保該 Key 存在且是 List
            if last_key not in target or not isinstance(target[last_key], list):
                target[last_key] = [] # 若不存在或格式錯，重置為空 List

            target_list = target[last_key]

            # 確保 List 長度足夠 (自動補齊)
            # 例如: 原本是 []，要更新 index = 2 -> 自動補成 [None, None, value]
            while len(target_list) <= index:
                target_list.append(None) # 或者補 0，看需求

            target_list[index] = value

        # 沒指定 index (覆寫/建立 List)
        else:
            if isinstance(value, list):
                target[last_key] = value
            else:
                dbg.war(f'data {value} is not list')
                target[last_key] = [value]

    def save_to_disk(self, file_id: JsonFileID):
        """
        將指定 ID 的記憶體資料寫入對應的硬碟檔案
        """
        # 檢查是否有註冊路徑
        file_path = self.path_map.get(file_id)
        if not file_path:
            dbg.error(f"無法存檔：{file_id} 沒有對應的檔案路徑")
            return

        # 從記憶體取得完整資料
        data = self.json_data.get(file_id)
        if data is None:
            dbg.war(f"存檔警告：{file_id} 記憶體為空")
            return

        # 寫入 (固定用覆蓋模式 w，因為記憶體已經是最新的)
        self._write_json(file_path, data, mode = 'w')

json_mg = JsonManager()
