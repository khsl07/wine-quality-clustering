@echo off
echo ========================================================
echo   Wine Quality Clustering & PCA System Setup
echo   Developed by khsl07
echo ========================================================

IF NOT EXIST "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate

echo [INFO] Installing dependencies...
pip install -r requirements.txt

echo.
echo [1] Run End-to-End Pipeline (python main.py)
echo [2] Launch Streamlit Interactive App (streamlit run app.py)
echo [3] Launch Jupyter Notebook (jupyter notebook)
echo.
set /p choice="Select option [1, 2, or 3]: "

if "%choice%"=="1" (
    python main.py
) else if "%choice%"=="2" (
    streamlit run app.py
) else if "%choice%"=="3" (
    jupyter notebook
) else (
    echo Invalid option. Exiting.
)

pause
