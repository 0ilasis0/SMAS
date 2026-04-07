@echo off
:: 切換到目前所在目錄
cd /d "%~dp0"

echo ========================================
echo 正在為您的 WinPython 安裝 pywebview...
echo ========================================

:: 使用精準路徑呼叫 pip 來安裝
.\python\WPy64-31241\python-3.12.4.amd64\python.exe -m pip install pywebview

echo.
echo ========================================
echo 安裝結束！請看上面是否有出現 Successfully installed 字樣。
echo ========================================
pause