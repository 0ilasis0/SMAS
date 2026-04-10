# info.filename → 呼叫程式所在檔案名稱
# info.lineno → 呼叫程式的行號
# info.function → 呼叫函式名稱
# info.code_context → 呼叫程式的原始程式碼
import datetime
import inspect
import os
import pprint
from contextlib import contextmanager


class Debug:
    def __init__(self, enable = True):
        self.enable = enable
        self.pp = pprint.PrettyPrinter(indent = 2, width = 100, sort_dicts = False)

    def _get_trace_string(self, start_frame):
        """
        取得過濾後的呼叫堆疊路徑 (包含資料夾名稱，並過濾掉系統檔案)
        格式: game_main.py:24(run) -> logic/manager.py:159(get_data)
        """
        # 取得當前工作目錄 (Project Root)
        project_root = os.getcwd()

        # 取得完整堆疊
        stack = inspect.getouterframes(start_frame)

        relevant_path_elements = []

        for frame_info in stack:
            abs_path = os.path.abspath(frame_info.filename)

            # 計算相對路徑 (這會自動包含資料夾，例如 'utils/manager.py')
            try:
                rel_path = os.path.relpath(abs_path, project_root)
            except ValueError:
                # 如果在不同磁碟機 (Windows)，relpath 會報錯，視為外部檔案
                continue

            # 過濾邏輯：
            # 如果路徑開頭是 ".."，代表檔案在專案資料夾外面 (例如 Python Lib)
            # 如果路徑包含 "site-packages"，代表是第三方套件
            # 如果是 IDE 的 debug 檔案 (通常包含 pydevd)
            if (rel_path.startswith("..") or
                "site-packages" in rel_path or
                "pydevd" in rel_path or
                "cli.py" in rel_path): # 針對你的案例過濾 cli.py

                # 一旦遇到系統層級的檔案，通常代表已經超出我們專案的範圍了
                # 我們可以選擇跳過，或者如果希望從 main 截斷，就不再繼續往上找
                continue

            # 組合顯示字串
            # 格式： 資料夾/檔名:行號(函式名)
            # 為了美觀，如果是 Windows 的反斜線 \ 可以換成 / (選用)
            display_path = rel_path.replace("\\", "/")

            lineno = frame_info.lineno
            func_name = frame_info.function

            relevant_path_elements.append(f"{display_path}:{lineno}({func_name})")

        # 因為 stack 是由內而外 (Current -> Main)，我們需要反轉變成 (Main -> Current)
        relevant_path_elements.reverse()

        return " -> ".join(relevant_path_elements)

    def dump(self, data, label = None):
        """
        專門用來印出複雜物件 (Dict, List, JSON)
        使用 Cyan 青色標記
        """
        if not self.enable: return

        trace_str = self._get_trace_string(inspect.currentframe().f_back)
        time = datetime.datetime.now().strftime("%H:%M:%S")

        # 使用 pformat 取得格式化後的字串
        formatted_data = self.pp.pformat(data)

        # 如果有標籤 (例如變數名稱)，加在前面
        prefix = f" ({label})" if label else ""

        # \033[96m 是青色 (Cyan)，適合區分一般 Log
        # 這裡加一個 \n 換行，讓物件內容從下一行開始，閱讀體驗較好
        print(f"\033[96m[DUMP {time} {trace_str}]{prefix}\033[0m\n{formatted_data}")

    def log(self, *args):
        if not self.enable: return
        trace_str = self._get_trace_string(inspect.currentframe().f_back)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\033[92m[DEBUG {time} {trace_str}]\033[0m", *args)

    def var(self, **kwargs):
        if not self.enable: return
        trace_str = self._get_trace_string(inspect.currentframe().f_back)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        for k, v in kwargs.items():
            print(f"\033[94m[VAR {time} {trace_str}]\033[0m {k} = {v}")

    def war(self, *args):
        if not self.enable: return
        trace_str = self._get_trace_string(inspect.currentframe().f_back)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\033[93m[WARNING {time} {trace_str}]\033[0m", *args)

    def error(self, *args):
        # Error 即使 disable 建議也要顯示，或者看你需求
        trace_str = self._get_trace_string(inspect.currentframe().f_back)
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\033[91m[ERROR {time} {trace_str}]\033[0m", *args)

    def toggle(self, forced: bool | None = None):
        if forced is not None:
            self.enable = forced
        else:
            self.enable = not self.enable

        print(f"\033[92m[DEBUG MODE: {self.enable}]\033[0m")

    @contextmanager
    def silence(self, active=True):
        """
        active=True: 執行靜音邏輯
        active=False: 什麼都不做，直接執行區塊內容
        """
        if not active:
            yield
            return

        previous_state = self.enable
        self.enable = False
        try:
            yield
        finally:
            self.enable = previous_state

dbg = Debug()
