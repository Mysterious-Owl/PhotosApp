@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create Python environment. Install Python 3.11 64-bit and enable the Python launcher.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Review the error above, then try install.bat again.
  pause
  exit /b 1
)
echo.
echo Install complete. Run run.bat to start Local Face Finder.
pause
