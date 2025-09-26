@echo off
REM Setup script for SPH Simulation project on Windows

echo 🚀 Setting up SPH Simulation project...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+ from https://python.org
    echo    Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ❌ requirements.txt not found. Please run this script from the project root directory.
    pause
    exit /b 1
)

echo 🐍 Setting up Python virtual environment...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install Python dependencies
echo 📚 Installing Python packages...
pip install -r requirements.txt

REM Verify installation
echo 🔍 Verifying installation...
python -c "import numpy as np; import matplotlib.pyplot as plt; import scipy; import moviepy; print('✅ All Python packages imported successfully!'); print(f'NumPy version: {np.__version__}'); print(f'Matplotlib version: {plt.matplotlib.__version__}'); print(f'SciPy version: {scipy.__version__}'); print(f'MoviePy version: {moviepy.__version__}')"

if errorlevel 1 (
    echo ❌ Package verification failed.
    pause
    exit /b 1
)

REM Check for FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  FFmpeg not found. Video generation may not work properly.
    echo    To install FFmpeg on Windows:
    echo    1. Download from https://ffmpeg.org/download.html
    echo    2. Extract and add to your system PATH
    echo    OR
    echo    3. Use chocolatey: choco install ffmpeg
    echo    4. Use winget: winget install ffmpeg
) else (
    echo ✅ FFmpeg is available
)

echo.
echo 🎉 Setup complete!
echo.
echo To activate the environment in the future, run:
echo    venv\Scripts\activate.bat
echo.
echo To run the simulations:
echo    cd 1 ^&^& python sodshock.py    # Sod shock tube simulation
echo    cd 2 ^&^& python planets.py     # Planetary simulation
echo.
pause