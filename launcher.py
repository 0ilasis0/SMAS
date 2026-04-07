import subprocess
import sys
import time

import webview


def start_streamlit():
    """在背景啟動 Streamlit 伺服器"""
    print("啟動 IDSS 量化引擎背景服務...")
    # 使用目前的 Python 執行檔啟動 Streamlit
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return process

if __name__ == '__main__':
    # 1. 啟動後端伺服器
    server_process = start_streamlit()

    # 2. 稍微等待 2~3 秒，確保 Streamlit 的 8501 port 已經啟動
    time.sleep(3)

    # 3. 建立原生的桌面視窗 (指向 localhost:8501)
    # 您可以在這裡自訂視窗大小、標題，甚至禁止使用者調整視窗大小
    window = webview.create_window(
        title='IDSS 台股量化交易終端',
        url='http://localhost:8501',
        width=1980,
        height=1080,
        min_size=(1400, 900)
    )

    # 4. 啟動視窗 (程式會停在這一行，直到使用者點擊右上角的 X 關閉視窗)
    webview.start()

    # ==========================================
    # 🛑 視窗關閉後的清理機制 (使用者點了 X)
    # ==========================================
    print("視窗已關閉，正在自動終止背景 AI 引擎...")
    server_process.terminate()  # 溫和地發送終止訊號
    server_process.wait()       # 等待進程完全關閉
    print("系統安全關閉完畢！")
    sys.exit(0)
