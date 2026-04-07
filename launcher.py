import os
import shutil
import subprocess
import sys
import tempfile
import time


def start_streamlit():
    print("啟動 IDSS 量化引擎背景服務...")
    # 將 stdout 和 stderr 導向 PIPE，避免在隱藏視窗模式 (pythonw) 下報錯
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def find_browser():
    """自動尋找客戶電腦中內建的 Edge 或 Chrome"""
    paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

if __name__ == '__main__':
    # 1. 啟動後端伺服器
    server_process = start_streamlit()
    time.sleep(3) # 等待伺服器暖機

    # 2. 尋找瀏覽器
    browser_path = find_browser()
    if not browser_path:
        print("❌ 找不到 Edge 或 Chrome 瀏覽器！")
        server_process.terminate()
        sys.exit(1)

    # 3. 建立獨立沙盒設定檔 (這是完美監聽視窗關閉的魔法！)
    temp_dir = os.path.join(tempfile.gettempdir(), "IDSS_App_Profile")

    print("啟動原生 App 視窗...")

    # 4. 啟動獨立的 App 視窗
    browser_process = subprocess.Popen([
        browser_path,
        "--app=http://localhost:8501",
        f"--user-data-dir={temp_dir}", # 強制獨立進程
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ==========================================
    # 🛑 程式會停在這裡，直到使用者點擊右上角的 X 關閉視窗
    # ==========================================
    browser_process.wait()

    # 5. 清理與關機
    print("視窗已關閉，正在自動終止背景 AI 引擎...")
    server_process.terminate()
    server_process.wait()

    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass

    print("系統安全關閉完畢！")
    sys.exit(0)