@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create a Python 3.12 environment first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
