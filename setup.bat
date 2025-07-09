@echo off
echo 🚀 Starting MedGemma 4B-IT POC Setup...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✅ Requirements installed successfully
echo.

REM Run setup script
echo 🔧 Running setup script...
python setup.py

echo.
echo 🎉 Setup completed!
echo.
echo 📋 To start the application:
echo 1. Run: venv\Scripts\activate.bat
echo 2. Run: streamlit run app.py
echo 3. Open your browser to the displayed URL
echo.
echo ⚠️  Important: Make sure you have accepted MedGemma terms on Hugging Face!
echo Visit: https://huggingface.co/google/medgemma-4b-it
echo.
pause
