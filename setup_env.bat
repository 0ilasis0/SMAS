@echo off
chcp 65001 >nul
echo ========================================================
echo   IDSS 專案環境一鍵安裝腳本 (WinPython 輕便版專用)
echo ========================================================

:: 設定相對路徑 (假設此 bat 檔放在 SMAS 根目錄下)
:: 根據您的圖片，真正的 python.exe 藏在底下的 python-3.12.4.amd64 資料夾中
set PYTHON_EXE=.\python\WPy64-31241\python-3.12.4.amd64\python.exe

:: 檢查 Python 執行檔是否存在
if not exist "%PYTHON_EXE%" (
    echo [錯誤] 找不到 Python 執行檔！
    echo 請確認您是將此 setup_env.bat 放在 SMAS 根目錄下。
    echo 預期路徑: %PYTHON_EXE%
    pause
    exit /b
)

echo [系統] 成功抓取到 WinPython 環境！準備進行套件安裝...
echo.

:: 1. 確保 pip 是最新版
echo ⏳ [1/2] 正在升級 pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

:: 2. 執行核心套件安裝
:: 註解：sklearn 正式名稱為 scikit-learn，dotenv 為 python-dotenv
:: google 套件則幫您配備最常用的 google-generativeai (Gemini) 與 api-client
echo ⏳ [2/2] 開始安裝專案依賴套件 (這可能需要幾分鐘，請耐心等候)...
echo.

"%PYTHON_EXE%" -m pip install seaborn pandas numpy matplotlib scikit-learn lightgbm xgboost streamlit python-dotenv yfinance plotly requests optuna tqdm torch google-generativeai google-api-python-client

echo.
echo ========================================================
echo   🎉 安裝完成！所有的套件都已安全鎖在輕便包內。
echo ========================================================
pause