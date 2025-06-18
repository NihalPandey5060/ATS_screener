@echo off
echo ========================================
echo    AI Resume Screening System
echo ========================================
echo.

echo Testing dependencies...
python test_installation.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Starting the application...
    echo The app will open in your default browser.
    echo If it doesn't open automatically, go to: http://localhost:8501
    echo.
    echo Press Ctrl+C to stop the application.
    echo.
    streamlit run resume_screener.py
) else (
    echo.
    echo Please install dependencies first:
    echo pip install -r requirements.txt
    echo.
    pause
) 