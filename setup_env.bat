@echo off
chcp 65001 >nul
echo ========================================================
echo   IDSS 專案環境一鍵安裝與自動更新腳本 (Requirements 讀取版)
echo ========================================================

:: 設定相對路徑 (假設此 bat 檔放在 SMAS 根目錄下)
set PYTHON_EXE=.\python\WPy64-31241\python-3.12.4.amd64\python.exe
set REQ_FILE=requirements.txt

:: 檢查 Python 執行檔是否存在
if not exist "%PYTHON_EXE%" (
    echo [錯誤] 找不到 Python 執行檔！
    echo 請確認您是將此 setup_env.bat 放在 SMAS 根目錄下。
    echo 預期路徑: %PYTHON_EXE%
    pause
    exit /b
)

:: 檢查 requirements.txt 是否存在
if not exist "%REQ_FILE%" (
    echo [錯誤] 找不到套件清單檔案 %REQ_FILE% ！
    echo 請確認該檔案與此 .bat 檔放在同一個目錄下。
    pause
    exit /b
)

echo [系統] 成功抓取到 WinPython 環境！準備進行套件安裝與更新...
echo.

:: 1. 確保 pip 是最新版
echo ⏳ [1/2] 正在升級 pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
echo.

:: 2. 讀取 requirements.txt 執行核心套件安裝與強制升級
echo ⏳ [2/2] 開始從 %REQ_FILE% 安裝並更新專案依賴套件...
echo 💡 (這可能需要幾分鐘，請耐心等候)...
echo.

"%PYTHON_EXE%" -m pip install --upgrade -r "%REQ_FILE%"

echo.
echo ========================================================
echo   🎉 安裝與更新完成！所有的套件都已依據 %REQ_FILE% 升級完畢。
echo ========================================================
pause