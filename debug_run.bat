@echo off
:: 切換到目前所在目錄
cd /d "%~dp0"

echo ========================================
echo 正在啟動 IDSS 除錯模式...
echo 觀察下方的錯誤訊息！
echo ========================================

:: 使用有黑窗的 python.exe 來執行 launcher.py (根據您第一張圖的路徑)
.\python\WPy64-31241\python-3.12.4.amd64\python.exe launcher.py

:: 暫停畫面，讓黑窗不會閃退，方便我們看錯誤訊息
pause