@echo off
chcp 65001 >nul
echo ========================================================
echo   IDSS 專案環境一鍵安裝與自動更新腳本 (安全鎖定版)
echo ========================================================

:: 設定相對路徑 (假設此 bat 檔放在 SMAS 根目錄下)
set PYTHON_EXE=.\python\WPy64-31241\python-3.12.4.amd64\python.exe

:: 檢查 Python 執行檔是否存在
if not exist "%PYTHON_EXE%" (
    echo [錯誤] 找不到 Python 執行檔！
    echo 請確認您是將此 setup_env.bat 放在 SMAS 根目錄下。
    echo 預期路徑: %PYTHON_EXE%
    pause
    exit /b
)

echo [系統] 成功抓取到 WinPython 環境！準備進行套件安裝與更新...
echo.

:: 1. 確保 pip 是最新版
echo ⏳ [1/2] 正在升級 pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

:: 2. 執行核心套件安裝與強制升級
:: 💡 已加入版本天花板鎖定 (numpy<2, pandas<3)，完美避開依賴地獄
echo ⏳ [2/2] 開始安裝並更新專案依賴套件 (這可能需要幾分鐘，請耐心等候)...
echo.

"%PYTHON_EXE%" -m pip install --upgrade seaborn "pandas<3.0.0" "numpy<2.0.0" matplotlib scikit-learn lightgbm xgboost streamlit python-dotenv yfinance plotly requests optuna tqdm "torch==2.2.2" google-generativeai google-api-python-client sqlalchemy typing-extensions "tenacity<9.0.0"

echo.
echo ========================================================
echo   🎉 安裝與更新完成！所有的套件都已升級至最安全版本。
echo ========================================================
pause