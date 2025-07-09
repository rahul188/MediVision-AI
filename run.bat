@echo off
echo 🚀 Starting MedGemma POC Application...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo ✅ Starting Streamlit application...
echo.
echo 📖 The application will open in your default browser
echo 🔧 Use Ctrl+C to stop the application
echo.

REM Start the Streamlit app
streamlit run app.py

pause
