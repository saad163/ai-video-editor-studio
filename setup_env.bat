@echo off
REM ============================================================
REM  AI Video Editor Studio — Windows Dev Environment Setup
REM  Run this script once from the project root directory.
REM  Usage: setup_env.bat
REM ============================================================

echo.
echo ============================================================
echo  AI Video Editor Studio - Environment Setup
echo ============================================================
echo.

REM Check Python 3.11+
python --version 2>NUL
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Install Python 3.11 from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/5] Creating virtual environment (.venv) ...
python -m venv .venv

REM Activate it
echo [2/5] Activating virtual environment ...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo [3/5] Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
echo [4/5] Installing project requirements ...
pip install -r requirements.txt

REM Copy .env template if .env doesn't exist
echo [5/5] Setting up .env file ...
IF NOT EXIST ".env" (
    copy ".env.example" ".env"
    echo [INFO] .env created from template. Add your GEMINI_API_KEY inside it.
) ELSE (
    echo [INFO] .env already exists — skipping copy.
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  TO START:
echo    .venv\Scripts\activate
echo    python main.py
echo.
echo  TO TEST TRANSCRIPTION (provide an audio file):
echo    python services\transcriber.py path\to\audio.mp3
echo.
echo  TO RUN UNIT TESTS:
echo    pytest tests\ -v
echo ============================================================
echo.
pause

