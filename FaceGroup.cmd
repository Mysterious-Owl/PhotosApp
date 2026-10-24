@echo off
rem Double-click to open FaceGroup. Keeps the Python environment outside OneDrive.
setlocal
cd /d "%~dp0"
set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\FaceGroup\venv"
python -m uv --version >nul 2>&1 || python -m pip install --user uv
python -m uv run facegroup ui %*
if errorlevel 1 pause
