@echo off
rem Start the edit ranking app (Windows: double-click).
rem The first run installs what the app needs into .venv (a few minutes, needs internet).
cd /d "%~dp0"

if exist ".venv\.installed" goto run
echo First run: installing the app. This takes a few minutes...
if exist ".venv" rmdir /s /q ".venv"
py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo Python 3.10 or newer is needed: https://www.python.org/downloads/
  echo When installing Python, tick "Add python.exe to PATH".
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installing failed. Check the internet connection and run this file again.
  pause
  exit /b 1
)
echo ok> ".venv\.installed"

:run
echo Starting the app. It opens in your browser; keep this window open while you rank.
".venv\Scripts\python.exe" app.py
pause
